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
      'onehot_norm_fast','fused_sum_fast','booth_fast','split_add_prefix_fast','retimed_sum_fast','retimed_sum_v2_fast','balanced_norm_fast')
PE_INTEGER=('pe_ctx_onehot','pe_ctx_both','pe_ctx_format2','int64_onehot_conversion',
            'int64_registered_prefix2','int64_prefix_sum2','int64_balanced_select')


def timing(path):
    data=json.loads(path.read_text());directory=path.parent
    identity=json.loads((directory/'build_identity.json').read_text())
    if data['rtl_sha256']!=identity['rtl_sha256'] or data['stages']!=2 or data['period_ns']!=1:
        raise ValueError('experimental RTL/stage/period mismatch: '+str(path))
    if data['mapped_sha256']!=sha(directory/'synth.v') or data['checkpoint_sha256']!=sha(directory/'post_cts.odb'):
        raise ValueError('experimental mapped netlist/checkpoint changed: '+str(path))
    return data


def run(out):
    cohort=verify(out/'functional/cohort.json')
    fp=[];pending=[];failed=[]
    for name in FP64:
        directory=ROOT/('.build/physical/asap7_TT_fp64_2_1x1_1ns_'+name)
        path=directory/'cts_timing.json'
        if not path.exists():
            log=directory/'elaborate.log'
            if (directory/'failure.json').exists()or(log.exists()and re.search(r'(?m)^ERROR:',log.read_text())):
                failed.append(dict(variant=name,elaboration=evidence(log),failure=evidence(directory/'failure.json')))
            else:pending.append(name)
            continue
        data=timing(path)
        numerical=directory/('rtl_verification.json'if(directory/'rtl_verification.json').exists()else'gate_verification.json')
        proof=json.loads(numerical.read_text())
        if not proof['passed']or proof['rtl_sha256']!=data['rtl_sha256']:
            raise ValueError('FP64 numerical verification differs: '+name)
        files=['cts_timing.json','build_identity.json',numerical.name]
        files += [n for n in('experiment_driver.py','booth_formal/verification.json')if(directory/n).is_file()]
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
    entries=[];pending=[]
    for name in PE_INTEGER:
        path=out/name/'verification.json'
        if not path.is_file():pending.append(name);continue
        record=json.loads(path.read_text())
        if 'timing'not in record:pending.append(name);continue
        item=record['timing'];p=Path(item['path'])
        if sha(p)!=item['sha256']:raise ValueError('experiment timing report changed: '+name)
        data=timing(p)
        if data!=item['data']or record['pipeline_stages']!=2 or not record['functional_passed']:
            raise ValueError('experiment functional/timing mismatch: '+name)
        for step in record['steps']:
            if step['returncode']or sha(step['log'])!=step['log_sha256']:
                raise ValueError('experiment failed or changed: '+name+'/'+step['name'])
        entries.append(dict(name=name,functional_passed=True,physical_frequency_certified=False,
            evidence=evidence(path),scope=data['scope'],rtl_sha256=data['rtl_sha256'],
            setup_wns_ns=data['setup_wns_ns'],hold_wns_ns=data['hold_wns_ns'],analysis='CTS placement estimate'))
    result=dict(main_source_changed=False,arithmetic_stages=2,target_core_period_ns=1,
        experiments=entries,pending_cts_variants=pending)
    (out/'pe_and_integer_optimization_history.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(fp64_completed=len(fp),pe_integer_completed=len(entries),physical_signoff=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/migration_asap7_f2')
    run(p.parse_args().out.resolve())
