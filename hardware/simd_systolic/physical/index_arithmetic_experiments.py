#!/usr/bin/env python3
"""Rebuild the experimental timing index from authenticated, separate results."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'physical')]
from freeze_verification import verify
from report_migration import evidence,sha

FP64=('raw_operands','carry_save_fast','zero_product_fast','prefix_pruned_fast',
      'onehot_norm_fast','fused_sum_fast','booth_fast','split_add_prefix_fast','retimed_sum_fast','retimed_sum_v2_fast','balanced_norm_fast','preinvert_cut_fast','onehot_norm_choices','preinvert_choices','speculative_finish_fast','chunked_cut8_fast','chunked_speculative_fast','predecoded_chunk_fast','split_four_fast','compact_exponent_fast','parallel_exponents_fast','effective_exponents_fast','effective_exponents_choices')
PE_INTEGER=('pe_ctx_onehot','pe_ctx_both','pe_ctx_format2','int64_onehot_conversion',
            'int64_registered_prefix2','int64_prefix_sum2','int64_balanced_select','int64_late_mask',
            'int64_carry_save_product','int64_carry_save_product3','int64_carry_save_product4','int64_baugh_product','int64_local_enable','int64_local_opcode','int64_binary_conversion',
            'int64_retimed_binary','int64_retimed_binary2')


def timing(path):
    data=json.loads(path.read_text());directory=path.parent
    identity=json.loads((directory/'build_identity.json').read_text())
    if data['rtl_sha256']!=identity['rtl_sha256'] or data['stages']!=2 or data['period_ns']!=1:
        raise ValueError('experimental RTL/stage/period mismatch: '+str(path))
    if data['mapped_sha256']!=sha(directory/'synth.v') or data['checkpoint_sha256']!=sha(directory/'post_cts.odb'):
        raise ValueError('experimental mapped netlist/checkpoint changed: '+str(path))
    if data.get('physical_driver_sha256')and data['physical_driver_sha256']!=sha(directory/'physical_driver.py'):
        archived=directory/('physical_driver_'+data['physical_driver_sha256'][:12]+'.py')
        if not archived.exists()or sha(archived)!=data['physical_driver_sha256']:
            raise ValueError('experimental physical driver snapshot changed: '+str(path))
    if data.get('mapping_profile')=='choices'and data['abc_script_sha256']!=sha(directory/'mapping_choices.abc'):
        raise ValueError('experimental ABC policy changed: '+str(path))
    for key,name in [('tool_input_netlist_sha256','tool_input.v'),('tool_input_formatter_sha256','netlist_formatter.py')]:
        if data.get(key)and data[key]!=sha(directory/name):raise ValueError('experimental import changed: '+name)
    return data


def run(out):
    cohort=verify(out/'functional/cohort.json')
    fp=[];pending=[];failed=[]
    for name in FP64:
        directory=ROOT/('.build/physical/asap7_TT_fp64_2_1x1_1ns_'+name)
        path=directory/'cts_timing.json'
        if not path.exists():
            log=directory/'elaborate.log'
            if any((directory/n).exists()for n in('failure.json','mapping_failure.json'))or(log.exists()and re.search(r'(?m)^ERROR:',log.read_text())):
                failed.append(dict(variant=name,elaboration=evidence(log),failure=evidence(directory/'failure.json'),
                    mapping_failure=evidence(directory/'mapping_failure.json')))
            else:pending.append(name)
            continue
        data=timing(path)
        if data.get('mapping_parent')and not(directory/'gate_verification.json').is_file():
            pending.append(name+'/mapped_numerical_check');continue
        if data.get('mapping_parent'):
            mapped=json.loads((directory/'gate_verification.json').read_text())
            if not mapped['passed']or mapped['netlist_sha256']!=data['mapped_sha256']or mapped['rtl_sha256']!=data['rtl_sha256']:
                raise ValueError('remapped FP64 numerical proof differs: '+name)
        numerical=directory/('rtl_verification.json'if(directory/'rtl_verification.json').exists()else'gate_verification.json')
        proof=json.loads(numerical.read_text())
        if not proof['passed']or proof['rtl_sha256']!=data['rtl_sha256']:
            raise ValueError('FP64 numerical verification differs: '+name)
        files=['cts_timing.json','build_identity.json',numerical.name]
        files += [n for n in('experiment_driver.py','booth_formal/verification.json','physical_driver.py',
            'mapping_choices.abc','mapping_experiment_driver.py','gate_verification.json',
            'parallel_exponent_driver.py','exponent_equivalence/verification.json',
            'exponent_native_widths1/verification.json')if(directory/n).is_file()]
        for relative in('exponent_equivalence','exponent_native_widths1'):
            p=directory/relative/'verification.json'
            if not p.exists():continue
            formal=json.loads(p.read_text())
            if not formal['passed']or sha(directory/relative/'exponents.sv')!=formal['primitive_sha256']:
                raise ValueError('FP exponent primitive proof differs')
            for check in formal['checks']:
                stem=f"e{check['width']}_b{check['bias']}"
                if not check['passed']or sha(directory/relative/(stem+'.log'))!=check['log_sha256']or sha(directory/relative/(stem+'.ys'))!=check['script_sha256']:
                    raise ValueError('FP exponent SAT evidence changed')
        fp.append(dict(variant=name,scope='FP64 component only',rtl_sha256=data['rtl_sha256'],
            stages=2,period_ns=1,setup_wns_ns=data['setup_wns_ns'],hold_wns_ns=data['hold_wns_ns'],
            analysis='CTS placement estimate; requires separate routed/extracted STA',
            integrated_into_main_design=False,physical_frequency_certified=False,
            evidence={n:evidence(directory/n)for n in files}))
    result=dict(target_core_period_ns=1,main_source_cohort=cohort['cohort_sha256'],
        main_design_changed_by_experiments=False,experiments=fp,pending_cts_variants=pending,
        failed_preflight_variants=failed,
        best_cts_setup_variant=max(fp,key=lambda e:e['setup_wns_ns'])['variant']if fp else None,
        all_variants_timing_closed=False,
        onehot_extracted_followup=evidence(out/'fp64_onehot_routed_diagnostic.json'))
    (out/'fp64_optimization_history.json').write_text(json.dumps(result,indent=2)+'\n')
    entries=[];pending=[];failed=[]
    for name in PE_INTEGER:
        path=out/name/'verification.json'
        if not path.is_file():pending.append(name);continue
        record=json.loads(path.read_text())
        bounded=out/name/'bounded_execution.json'
        if bounded.exists():
            execution=json.loads(bounded.read_text())
            if sha(Path(execution['execution_log']))!=execution['log_sha256']:
                raise ValueError('bounded experiment log changed: '+name)
            if execution['timed_out']or execution['returncode']:
                failed.append(dict(name=name,functional_passed=record['functional_passed'],
                    failed_steps=['bounded_execution'],evidence=evidence(path),execution=evidence(bounded)))
                continue
        if 'timing'not in record:
            failures=[s for s in record['steps']if s['returncode']]
            for step in record['steps']:
                if sha(step['log'])!=step['log_sha256']:raise ValueError('experiment log changed: '+name)
            if failures:failed.append(dict(name=name,functional_passed=record['functional_passed'],
                failed_steps=[s['name']for s in failures],evidence=evidence(path)))
            else:pending.append(name)
            continue
        item=record['timing'];p=Path(item['path'])
        if sha(p)!=item['sha256']:raise ValueError('experiment timing report changed: '+name)
        data=timing(p)
        if data!=item['data']or record['pipeline_stages']!=2 or not record['functional_passed']:
            raise ValueError('experiment functional/timing mismatch: '+name)
        for step in record['steps']:
            if step['returncode']or sha(step['log'])!=step['log_sha256']:
                raise ValueError('experiment failed or changed: '+name+'/'+step['name'])
        mapped_proof=p.parent/'check_mapped/verification.json'
        if mapped_proof.exists():
            proof=json.loads(mapped_proof.read_text())
            if not proof['passed']or proof['rtl_sha256']!=data['rtl_sha256']or proof['netlist_sha256']!=data['mapped_sha256']:
                raise ValueError('experiment mapped numerical evidence differs: '+name)
            if proof.get('simulated_netlist_sha256')and proof['simulated_netlist_sha256']!=sha(p.parent/'tool_input.v'):
                raise ValueError('experiment simulated another tool representation: '+name)
        entries.append(dict(name=name,functional_passed=True,physical_frequency_certified=False,
            mapped_numerical=evidence(mapped_proof),
            evidence=evidence(path),scope=data['scope'],rtl_sha256=data['rtl_sha256'],
            setup_wns_ns=data['setup_wns_ns'],hold_wns_ns=data['hold_wns_ns'],analysis='CTS placement estimate'))
    result=dict(main_source_changed=False,arithmetic_stages=2,target_core_period_ns=1,
        experiments=entries,pending_cts_variants=pending,failed_variants=failed)
    followups=[]
    for name in('carry_save_product4_wrapped','carry_save_product4_scalar','baugh_product_scalar'):
        directory=ROOT/('.build/physical/asap7_TT_int64_2_1x1_1ns_'+name)
        path=directory/'cts_timing.json';proof_path=directory/'check_mapped/verification.json'
        item=dict(name=name,timing=evidence(path),mapped_numerical=evidence(proof_path),
            failure=evidence(directory/'failure.json'),physical_frequency_certified=False)
        if path.exists():
            data=timing(path);item.update(setup_wns_ns=data['setup_wns_ns'],hold_wns_ns=data['hold_wns_ns'])
        if proof_path.exists():
            proof=json.loads(proof_path.read_text());identity=json.loads((directory/'build_identity.json').read_text())
            if not proof['passed']or proof['rtl_sha256']!=identity['rtl_sha256']or proof['netlist_sha256']!=sha(directory/'synth.v'):
                raise ValueError('mapped I64 proof differs from imported implementation')
            if proof.get('simulated_netlist_sha256')!=sha(directory/'tool_input.v'):
                raise ValueError('mapped I64 simulation did not use the actual imported netlist')
            item['mapped_numerical_passed']=True
        followups.append(item)
    result['mapped_import_followups']=followups
    routed=[]
    for name in('local_enable_routed','local_enable_routed_resume1','local_enable_routed_resume2',
            'local_enable_checkpointed_geometry','local_enable_checkpointed_extracted',
            'observable_cleanup1_geometry','baugh_observable_cleanup1_geometry','baugh_local_enable1_geometry'):
        directory=ROOT/('.build/physical/asap7_TT_int64_2_1x1_1ns_'+name)
        parent_path=directory/'route_parent.json'
        if not parent_path.is_file():continue
        parent=json.loads(parent_path.read_text());source=Path(parent['source'])
        for path,digest in [(source/'cts_timing.json',parent['source_cts_sha256']),
                (Path(parent['source_mapped_numerical']),parent['source_mapped_numerical_sha256']),
                (directory/'synth.v',parent['mapped_sha256'])]:
            if sha(path)!=digest:raise ValueError('routed arithmetic parent changed: '+str(path))
        item=dict(name=name,source=evidence(parent_path),physical_frequency_certified=False,
            execution=evidence(directory/'route_execution.json'),timing=evidence(directory/'timing.json'),
            mapped_parent_numerical=evidence(Path(parent['source_mapped_numerical'])),
            status='pending')
        execution=directory/'route_execution.json'
        if execution.is_file():
            record=json.loads(execution.read_text())
            if sha(directory/'execution.log')!=record['execution_log_sha256']or sha(directory/'route.log')!=record['route_log_sha256']:
                raise ValueError('routed arithmetic execution log changed')
            if record.get('detailed_route_only') and record.get('detailed_route_checkpoint_sha256'):
                checkpoint=directory/'detailed_route_checkpoint.json'
                if sha(checkpoint)!=record['detailed_route_checkpoint_sha256']:
                    raise ValueError('detailed-route geometry checkpoint changed')
                for path,digest in json.loads(checkpoint.read_text())['evidence'].items():
                    if sha(path)!=digest:raise ValueError('detailed-route checkpoint evidence changed: '+path)
                item.update(status='geometry_saved_extraction_pending',geometry_checkpoint=evidence(checkpoint))
            else:item['status']='completed'if record['routed_sta_completed']else'failed_or_timed_out'
        timing_path=directory/'timing.json'
        if timing_path.is_file():
            data=json.loads(timing_path.read_text())
            if data['rtl_sha256']!=parent['rtl_sha256']or data['stages']!=2 or data['period_ns']!=1:
                raise ValueError('routed arithmetic changed implementation/stages/period')
            for label,path in data['evidence'].items():
                if sha(path)!=data['evidence_sha256'][label]:raise ValueError('routed arithmetic evidence changed: '+label)
            if sha(directory/'flow.tcl')!=data['flow_sha256']:raise ValueError('routed arithmetic flow changed')
            item.update(setup_wns_ns=data['setup_wns_ns'],hold_wns_ns=data['hold_wns_ns'],tns_ns=data['tns_ns'],
                electrical_clean=data['electrical_clean'],power_grid_connectivity_verified=data.get('power_grid_connectivity_verified',False))
            from parasitics import audit as audit_parasitics
            annotation=audit_parasitics((directory/'route.log').read_text(),directory/'design.spef')
            item['parasitic_annotation_clean']=annotation['clean']
            if not annotation['clean']:
                item.update(status='unqualified_parasitic_annotation',parasitic_problems=annotation['problems'])
            proof_path=directory/'check_routed/verification.json'
            if proof_path.is_file():
                proof=json.loads(proof_path.read_text())
                if not proof['passed']or not proof['routed_netlist']or proof['netlist_sha256']!=data['netlist_sha256']or proof['rtl_sha256']!=data['rtl_sha256']:
                    raise ValueError('routed arithmetic numerical check differs')
                item['routed_numerical']=evidence(proof_path)
        routed.append(item)
    result['routed_followups']=routed
    recoveries=[]
    for name in('safe_names1','observable_cleanup1_export','baugh_observable_cleanup1_export','baugh_local_enable1_export'):
        directory=ROOT/('.build/physical/asap7_TT_int64_2_1x1_1ns_'+name)
        state_path=directory/'recovery_state.json'
        if not state_path.exists():continue
        state=json.loads(state_path.read_text())
        item=dict(name=name,status='running'if state.get('active_phase')else'completed'if state['passed']else'failed',
            verification=evidence(directory/'recovery_verification.json'),physical_frequency_certified=False)
        if(directory/'recovery_verification.json').exists():
            if state['arithmetic_stages']!=2 or state['core_period_ns']!=1:raise ValueError('recovery changed stages or period')
            parent=Path(state['source_checkpoint'])
            if sha(parent)!=state['source_checkpoint_sha256']:raise ValueError('recovered geometry parent changed')
            for step in state['steps']:
                if sha(Path(step['log']))!=step['log_sha256']:raise ValueError('recovery execution log changed')
            item['parent']=evidence(parent)
            path=directory/'timing.json'
            if path.exists():
                data=json.loads(path.read_text())
                for label,filename in data['evidence'].items():
                    if sha(filename)!=data['evidence_sha256'][label]:raise ValueError('recovery STA evidence changed')
                item.update(timing=evidence(path),setup_wns_ns=data['setup_wns_ns'],hold_wns_ns=data['hold_wns_ns'],
                    parasitic_annotation_clean=data.get('parasitic_annotation_clean',False))
            item['routed_numerical']=evidence(directory/'check_routed/verification.json')
        recoveries.append(item)
    result['routed_export_recoveries']=recoveries
    rechecks=[]
    for variant in('safe_names_qualified1','observable_cleanup1_qualified','baugh_observable_cleanup1_qualified','baugh_local_enable1_qualified'):
        directory=ROOT/('.build/physical/asap7_TT_int64_2_1x1_1ns_'+variant)
        path=directory/'recheck_verification.json'
        if not path.exists():continue
        record=json.loads(path.read_text())
        if record['stages']!=2 or record['core_period_ns']!=1:raise ValueError('STA recheck changed target configuration')
        for filename,digest in record['evidence'].items():
            if sha(filename)!=digest:raise ValueError('STA recheck evidence changed: '+filename)
        data=json.loads((directory/'timing.json').read_text())
        for name,filename in data['evidence'].items():
            if sha(filename)!=data['evidence_sha256'][name]:raise ValueError('qualified STA evidence changed: '+name)
        rechecks.append(dict(name=variant,evidence=evidence(path),
            setup_wns_ns=record.get('setup_wns_ns'),hold_wns_ns=record.get('hold_wns_ns'),
            tns_ns=record.get('tns_ns'),parasitic_annotation_clean=record.get('parasitic_annotation_clean',False),
            routed_numerical_passed=record['routed_numerical_passed'],setup_hold_closed=record.get('setup_hold_closed',False),
            physical_frequency_certified=False))
    result['routed_sta_rechecks']=rechecks
    clock_experiments=[]
    directory=ROOT/'.build/physical/asap7_TT_int64_2_1x1_1ns_local_enable_clustered64'
    path=directory/'physical_experiment_parent.json'
    if path.exists():
        record=json.loads(path.read_text())
        if sha(record['source_numerical'])!=record['source_numerical_sha256']:
            raise ValueError('clock experiment numerical parent changed')
        item=dict(name='local_enable_clustered64',parent=evidence(path),
            execution=evidence(directory/'physical_experiment_execution.json'),physical_frequency_certified=False)
        if(directory/'cts_timing.json').exists():
            data=timing(directory/'cts_timing.json')
            item.update(setup_wns_ns=data['setup_wns_ns'],hold_wns_ns=data['hold_wns_ns'],
                analysis='CTS placement estimate, not extracted STA')
        clock_experiments.append(item)
    result['clock_placement_experiments']=clock_experiments
    cleanups=[]
    for variant,cts_variant in(('observable_cleanup1','observable_cleanup1_cts'),
            ('baugh_observable_cleanup1','baugh_observable_cleanup1'),('four_term_cut1','four_term_cut1'),
            ('baugh_local_enable1','baugh_local_enable1')):
        directory=ROOT/('.build/physical/asap7_TT_int64_2_1x1_1ns_'+variant)
        proof_path=directory/'observable_cleanup.json'
        if not proof_path.exists():continue
        proof=json.loads(proof_path.read_text());identity=json.loads((directory/'build_identity.json').read_text())
        if not proof['passed']or(proof['pipeline_stages'],proof['core_period_ns'])!=(2,1):
            raise ValueError('observable cleanup has no exact structural certificate')
        if proof['mapped_sha256']!=sha(directory/'synth.v')or identity['mapped_sha256']!=proof['mapped_sha256']:
            raise ValueError('observable cleanup mapped identity changed')
        source=Path(proof['source'])
        for path,digest in [(source/'build_identity.json',proof['source_identity_sha256']),
                (source/'synth.v',proof['source_mapped_sha256'])]+[(Path(p),v)for p,v in proof['evidence'].items()]:
            if sha(path)!=digest:raise ValueError('observable cleanup evidence changed: '+str(path))
        item=dict(name=variant,structural_certificate=evidence(proof_path),
            modules=proof['modules'],stages=2,period_ns=1,retiming_performed=False,
            physical_frequency_certified=False,main_source_changed=False,
            mapped_numerical_passed=False,status='awaiting_mapped_numerical_check')
        numerical_path=directory/'check_mapped/verification.json'
        if numerical_path.exists():
            numerical=json.loads(numerical_path.read_text())
            if not numerical['passed']or numerical['netlist_sha256']!=proof['mapped_sha256']or numerical['rtl_sha256']!=identity['rtl_sha256']:
                raise ValueError('observable cleanup numerical identity changed')
            if numerical['arithmetic_cycles_without_stalls']!=2:
                raise ValueError('observable cleanup changed arithmetic latency')
            imported_checked=numerical['simulated_netlist_sha256']==sha(directory/'tool_input.v')
            item.update(mapped_numerical_passed=True,mapped_numerical=evidence(numerical_path),
                numerical_checks=numerical['checks'],actual_import_numerical_passed=imported_checked,
                status='mapped_numerical_passed'if imported_checked else'awaiting_actual_import_numerical_check')
        cts_directory=ROOT/('.build/physical/asap7_TT_int64_2_1x1_1ns_'+cts_variant)
        cts_path=cts_directory/'cts_timing.json'
        if cts_path.exists():
            data=timing(cts_path)
            if data['mapped_sha256']!=proof['mapped_sha256']:
                raise ValueError('observable cleanup CTS has another mapped netlist')
            item.update(cts=evidence(cts_path),setup_wns_ns=data['setup_wns_ns'],
                hold_wns_ns=data['hold_wns_ns'],analysis='CTS placement estimate; requires extracted STA')
        cleanups.append(item)
    result['observable_mapped_cleanups']=cleanups
    cuts=[]
    for sequence,terms in(('int64_four_term_cut1',4),('int64_four_term_recovery1',4),
            ('int64_baugh_local_enable1',2)):
        directory=out/sequence;path=directory/'state.json'
        if not path.exists():continue
        record=json.loads(path.read_text())
        if(record['arithmetic_stages'],record['core_period_ns'],record['first_boundary_product_terms'])!=(2,1,terms):
            raise ValueError('product-cut experiment changed architecture constraints')
        for step in record['steps']:
            if sha(step['log'])!=step['log_sha256']:raise ValueError('product-cut execution log changed')
        item=dict(name=sequence,state=evidence(path),
            completed=record['passed'],active_phase=record.get('active_phase'),
            error=record.get('error'),failed_steps=[s['name']for s in record['steps']if s['returncode']or s['timed_out']],
            functional_passed=record.get('functional_passed',False),
            physical_frequency_certified=False,main_source_changed=False)
        functional=directory/'functional/verification.json'
        if functional.exists():
            proof=json.loads(functional.read_text())
            if proof['pipeline_stages']!=2 or proof['integer_product_terms']!=terms:
                raise ValueError('product-cut functional proof changed pipeline')
            for step in proof['steps']:
                if sha(step['log'])!=step['log_sha256']:
                    raise ValueError('product-cut functional step changed')
                if step['returncode']and proof['functional_passed']:
                    raise ValueError('failed functional step was relabeled as passed')
            item.update(functional=evidence(functional),functional_passed=proof['functional_passed'])
        cuts.append(item)
    result['product_cut_experiments']=cuts
    formal=ROOT/'.build/experiments/int64_coi_analysis1/formal_execution.json'
    if formal.exists():
        record=json.loads(formal.read_text())
        result['general_equivalence_trial']=dict(evidence=evidence(formal),passed=record['passed'],
            timed_out=record['timed_out'],independent_of_exact_structural_certificate=True)
    pe_followups=[]
    for name in('best_local_controls','best_local_controls_scalar'):
        directory=ROOT/('.build/physical/asap7_TT_pe_2_1x1_1ns_'+name)
        parent_path=directory/'experiment_parent.json'
        if not parent_path.is_file():continue
        parent=json.loads(parent_path.read_text());proof=Path(parent['source_functional_verification'])
        if sha(proof)!=parent['source_functional_verification_sha256']or not json.loads(proof.read_text())['functional_passed']:
            raise ValueError('PE followup numerical source changed')
        if parent.get('mapped_retry_source')and sha(Path(parent['mapped_retry_source'])/'synth.v')!=parent['mapped_sha256']:
            raise ValueError('PE mapped retry changed the source netlist')
        item=dict(name=name,parent=evidence(parent_path),execution=evidence(directory/'execution.json'),
            status='pending',physical_frequency_certified=False)
        execution_path=directory/'execution.json'
        if execution_path.is_file():
            execution=json.loads(execution_path.read_text())
            if sha(Path(execution['execution_log']))!=execution['execution_log_sha256']:
                raise ValueError('PE followup execution log changed')
            item['status']='failed_or_timed_out'if execution['returncode']or execution['timed_out']else'completed'
        path=directory/'cts_timing.json'
        if path.is_file():
            data=timing(path)
            item.update(timing=evidence(path),setup_wns_ns=data['setup_wns_ns'],hold_wns_ns=data['hold_wns_ns'])
        pe_followups.append(item)
    result['pe_physical_followups']=pe_followups
    (out/'pe_and_integer_optimization_history.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(fp64_completed=len(fp),pe_integer_completed=len(entries),physical_signoff=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/migration_asap7_f2')
    run(p.parse_args().out.resolve())
