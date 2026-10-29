#!/usr/bin/env python3
"""Collect authenticated migration evidence; never infer physical signoff.

Safe to run while a regression is active: pending steps remain pending.
This report deliberately keeps digital-model acceptance and physical frequency
acceptance separate, including for the generated 6T research SRAM.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'),str(ROOT/'python'),str(ROOT/'physical')]
from freeze_verification import verify
from simd_eval.architecture import ARCH
from sram_package import load_package


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def evidence(path):
    return dict(path=str(path.resolve()),sha256=sha(path)) if path.is_file() else None


def suite(directory,passed_key,required_count):
    path=directory/'acceptance.json';data=read(path)
    if data is None:
        return dict(status='pending',passed=False,steps=0,evidence=None)
    for step in data['steps']:
        if sha(step['log'])!=step['log_sha256']:
            raise ValueError('regression log changed: '+step['name'])
    failed=[s['name'] for s in data['steps'] if s['returncode']]
    complete=(data.get(passed_key) is True and not failed and
        len(data['steps'])==required_count and len({s['name'] for s in data['steps']})==required_count)
    if data.get(passed_key) and not complete:
        raise ValueError('incomplete or duplicate required regression steps')
    return dict(status='passed' if complete else 'failed' if failed else 'pending',
        passed=complete,steps=len(data['steps']),required_steps=required_count,
        failed_steps=failed,evidence=evidence(path))


def run(out):
    cohort=verify(out/'functional/cohort.json')
    tool_tests=read(out/'physical_tool_tests.json')
    if tool_tests:
        if sha(tool_tests['log'])!=tool_tests['log_sha256']:
            raise ValueError('physical tool test log changed')
        for name,digest in tool_tests['sources_sha256'].items():
            if sha(ROOT/name)!=digest:raise ValueError('physical tool test source changed: '+name)
    package=load_package(out/'sram_openfinram',ARCH)
    base=read(out/'functional/acceptance.json')
    native=read(out/'generated_sram_gemm/acceptance.json')
    if base['cohort_sha256']!=cohort['cohort_sha256']:
        raise ValueError('functional suite source cohort changed')
    if native and (native['source_cohort_sha256']!=cohort['cohort_sha256'] or
            native['macro_package_manifest_sha256']!=package['manifest_sha256']):
        raise ValueError('generated macro regression identity changed')
    regular=suite(out/'functional','functional_passed',57)
    generated=suite(out/'generated_sram_gemm','functional_model_passed',14)
    derived_path=out/'sram_openfinram_power_access2'
    derived=None;power=None
    if (derived_path/'package.json').exists():
        derived=load_package(derived_path,ARCH)
        if (sha(derived['wrapper'])!=sha(package['wrapper']) or
                sha(derived['root']/(derived['native']+'.sp'))!=sha(package['root']/(package['native']+'.sp'))):
            raise ValueError('power revision changed the logical model or ideal transistor schematic')
        power=read(out/'sram_power_probe5/verification.json')
        if power:
            if power['macro_package_manifest_sha256']!=derived['manifest_sha256']:
                raise ValueError('power probe uses another macro abstract')
            for item in power['evidence'].values():
                if sha(item['path'])!=item['sha256']:raise ValueError('power probe evidence changed')
    latest_path=out/'sram_openfinram_eco_access'
    latest=None;controller=None;controller_corners={}
    if (latest_path/'package.json').exists():
        latest=load_package(latest_path,ARCH)
        if sha(latest['wrapper'])!=sha(package['wrapper']):raise ValueError('latest controller ECO changed the logical SRAM model')
        integration=read(latest_path/'controller_integration.json')
        if not integration or not integration['controller_integrated']or not integration['controller_sequential_equivalence']:
            raise ValueError('latest macro is missing controller integration/equivalence evidence')
        controller=read(out/'sram_controller_arch_tt25_v3/verification.json')
        power=read(out/'sram_power_probe6/verification.json')
        if power:
            if power['macro_package_manifest_sha256']!=latest['manifest_sha256']:raise ValueError('latest power evidence uses another macro')
            for item in power['evidence'].values():
                if sha(item['path'])!=item['sha256']:raise ValueError('latest power probe evidence changed')
        if controller:
            for path,digest in controller['evidence'].items():
                if sha(path)!=digest:raise ValueError('controller STA evidence changed')
            matching=[digest for path,digest in controller['evidence'].items()if path.endswith('/controller_eco.odb')]
            if matching!=[sha(latest_path/'controller_eco/controller_eco.odb')]:raise ValueError('controller STA does not match the integrated checkpoint')
        for corner,path in [('TT',out/'sram_controller_arch_tt25_v3/verification.json'),
                ('SS',out/'sram_controller_arch_ss/verification.json'),
                ('FF',out/'sram_controller_arch_ff/verification.json')]:
            record=read(path)
            if record is None:continue
            for item,digest in record['evidence'].items():
                if sha(item)!=digest:raise ValueError('controller corner evidence changed: '+item)
            checkpoint=[digest for item,digest in record['evidence'].items()if item.endswith('/controller_eco.odb')]
            if checkpoint!=[sha(latest_path/'controller_eco/controller_eco.odb')]:raise ValueError('controller corners use different physical checkpoints')
            if record.get('process_corner',corner)!=corner or record['period_ns']!=2:
                raise ValueError('controller corner/period differs')
            controller_corners[corner]=dict(evidence=evidence(path),
                voltage_V=record['pvt_voltage_V'],temperature_C=record['pvt_temperature_C'],
                setup_wns_ns=record['fixed_mode_setup_wns_ns'],hold_wns_ns=record['fixed_mode_hold_wns_ns'],
                dynamic_endpoints_covered=record['coverage']['dynamic_endpoints_covered'],
                controller_only=True,whole_macro_characterized=False)
    native8=read(out/'generated_sram_gemm8/verification.json')
    if native8:
        if native8['source_cohort_sha256']!=cohort['cohort_sha256']:raise ValueError('native8 source cohort differs')
        for item in native8['evidence'].values():
            if sha(item['path'])!=item['sha256']:raise ValueError('native8 evidence changed')
    bitcell=read(out/'sram_bitcell_transistor/verification.json')
    if bitcell:
        if not latest or bitcell['macro_package_manifest_sha256']!=latest['manifest_sha256']:
            raise ValueError('bitcell evidence uses another macro schematic')
        for path,digest in bitcell['evidence'].items():
            if sha(path)!=digest:raise ValueError('bitcell transistor evidence changed')
        if bitcell['passed']and(len(bitcell['operating_points'])!=3 or any(
                len(r['checks'])!=8 or not r['passed']for r in bitcell['operating_points'])):
            raise ValueError('incomplete three-corner bitcell check')
    full_grid=read(out/'system_floorplan_eco4/verification.json')
    if full_grid:
        if not latest or full_grid['physical_macro_package_manifest_sha256']!=latest['manifest_sha256']:
            raise ValueError('system floorplan uses another physical macro revision')
        for path,digest in full_grid['evidence'].items():
            if sha(path)!=digest:raise ValueError('system floorplan evidence changed')
    protected_power=read(out/'system_floorplan_protected_power2/verification.json')
    if protected_power:
        if not full_grid or protected_power['source_verification_sha256']!=sha(out/'system_floorplan_eco4/verification.json'):
            raise ValueError('protected supply repair has another parent grid')
        if protected_power['physical_macro_package_manifest_sha256']!=latest['manifest_sha256']:
            raise ValueError('protected supply repair has another physical macro')
        for path,digest in protected_power['evidence'].items():
            if sha(path)!=digest:raise ValueError('protected supply repair evidence changed: '+path)
    groups={
        'two_stage_arithmetic':['alu','alu_stall','alu_idle_resume','f2_reference'],
        'pe_array':['check_pe','check_pe_state','check_ii','check_array','check_array_events',
            'array_cycles','array_hazards']+[f'long_{fmt}_{k}' for fmt in ('I32','I64') for k in (1,7,8,9,65535,65536)],
        'sram_capacity_masks':['sram_bank','scratch2','scratch8'],
        'dma_dram':['dram_adapter','faults2','faults8','check_epilogue2','check_epilogue8'],
        'multi_clock':['phase'+str(p) for p in (0,250,500,1500)]+[
            'reset'+str(rows)+'_'+domain for rows in (2,8) for domain in ('both','core','memory')]+['cdc2','cdc8'],
        'end_to_end':['auxiliary','check_descriptor2','check_descriptor8','check_grouped2',
            'check_grouped8','prefetch2','prefetch8'],
        'infrastructure':['unit_tests']}
    completed={s['name']:s for s in base['steps']}
    categories={}
    for name,names in groups.items():
        failed=[n for n in names if n in completed and completed[n]['returncode']]
        pending=[n for n in names if n not in completed]
        categories[name]=dict(status='failed' if failed else 'pending' if pending else 'passed',
            required_steps=names,pending_steps=pending,failed_steps=failed)
    matrix=dict(schema_version=1,architecture=ARCH,
        source_cohort_sha256=cohort['cohort_sha256'],
        generated_macro_package_manifest_sha256=package['manifest_sha256'],
        stages={
            'M0':dict(status='implemented',evidence=[evidence(out/'baseline/manifest.json'),
                evidence(out/'baseline/historical_reports_index.json'),evidence(out/'functional/cohort.json'),
                evidence(out/'implementation/source_manifest.json'),evidence(out/'change_inventory.json')]),
            'M1':dict(status='functional_passed_physical_incomplete' if regular['passed'] else 'functional_pending',
                digital=regular,core_frequency_certified=False,
                isolated_context_and_fpu_experiment=evidence(out/'pe_ctx_onehot/verification.json'),
                isolated_all_context_reads_experiment=evidence(out/'pe_ctx_both/verification.json'),
                isolated_local_format_experiment=evidence(out/'pe_ctx_format2/verification.json'),
                frozen_pe_cts_diagnostic=evidence(out/'qualified_pe_diagnostic.json'),
                int64_component=evidence(out/'int64_component_diagnostic.json'),
                integer_conversion_experiments=[evidence(out/'int64_onehot_conversion/verification.json'),
                    evidence(out/'int64_onehot_conversion/rounding_sweep/verification.json'),
                    evidence(out/'int64_registered_prefix2/verification.json'),
                    evidence(out/'int64_prefix_sum2/verification.json')],
                arithmetic_experiment_histories=[evidence(out/'fp64_optimization_history.json'),
                    evidence(out/'pe_and_integer_optimization_history.json')],
                remaining=['Current FP64 extracted setup/electrical failures',
                    'INT64, complete PE and array extracted timing at 1ns; matching SS/FF corners']),
            'M2':dict(status='6t_research_prototype_constructed',
                evidence=evidence(out/'sram_openfinram/package.json'),
                power_access_package=evidence(derived_path/'package.json'),
                latest_controller_eco_package=evidence(latest_path/'package.json'),
                unchanged_logical_model=derived is not None,
                controller_fixed_mode_architecture_TT=controller,
                controller_corner_diagnostics=controller_corners,
                bitcell_schematic_transistor_diagnostic=evidence(out/'sram_bitcell_transistor/verification.json'),
                bitcell_three_corner_checks_passed=bool(bitcell and bitcell['passed']),
                full_macro_schematic_experiment=evidence(out/'sram_macro_transistor_tt/verification.json'),
                full_macro_operating_point_retry=evidence(out/'sram_macro_transistor_tt_op/verification.json'),
                reduced_column_convergence_diagnostics=[evidence(out/name/'verification.json')for name in(
                    'sram_column_transistor_tt_op','sram_column_transistor_tt_op2','sram_column_transistor_tt_shunt')],
                active_bitcells_per_macro=65536,macro_instances=16,
                physical_certified=False,frequency_certified=False,
                remaining=['Full macro DRC/LVS/PEX',
                    'Full native macro transistor verification across addresses and corners',
                    'Measured TT/SS/FF Liberty characterization']),
            'M3':dict(status='digital_model_passed' if regular['passed'] and generated['passed'] else 'digital_model_pending',
                default_macro_interface=regular,generated_macro_2x2=generated,
                generated_macro_8x8=evidence(out/'generated_sram_gemm8/verification.json'),
                note='Full57-step suite uses the frozen logical interface; native-pin wrapper additionally passed14 checks in2x2 and all-format descriptor smoke/full128KiB SRAM sweep in8x8. Each report keeps its actual macro package identity.'),
            'M4':dict(status='incomplete',physical_signoff=False,
                macro_only_power_connected=bool(power and power['grid_generation_and_connectivity_passed']),
                macro_power_scope=power['grid_scope']if power else None,
                mapped_system_full_row_grid=evidence(out/'system_floorplan_eco4/verification.json'),
                mapped_system_floorplan_power_connected=bool(full_grid and full_grid['grid_generation_and_connectivity_passed']),
                protected_system_supply_repair=evidence(out/'system_floorplan_protected_power2/verification.json'),
                protected_system_supplies_connected=bool(protected_power and protected_power['passed']),
                physical_tool_tests=evidence(out/'physical_tool_tests.json'),
                evidence=[evidence(out/'physical_diagnostic.json'),
                    evidence(out/'generated_sram_system_diagnostic.json'),
                    evidence(out/'generated_sram_gemm_diagnostic.json'),
                    evidence(out/'generated_sram_gemm8_diagnostic.json'),
                    evidence(out/'sram_power_probe6/verification.json'),
                    evidence(out/'sram_controller_arch_tt25_v3/verification.json'),
                    evidence(out/'fp64_onehot_routed_diagnostic.json'),
                    evidence(out/'fp64_optimization_history.json')],
                remaining=['Complete integrated standard-cell and macro power implementation',
                    'Whole PE, 2x2 and 8x8 system placement/routing/extraction',
                    'Setup/hold/TNS/electrical/coverage/route checks at 1ns/2ns',
                    'Qualified SRAM internal evidence and matched PVT corner checks'])},
        functional_categories=categories,
        functional_model_acceptance=regular['passed'] and generated['passed'],
        full_hardware_plan_complete=False,physical_frequency_certified=False,
        note='Constructed 6T geometry and a logical model do not certify transistor behavior or operating frequency.',
        report_generator=evidence(Path(__file__)))
    (out/'acceptance_matrix.json').write_text(json.dumps(matrix,indent=2,ensure_ascii=False)+'\n')
    fields=('stages','rows','cols','descriptors','formats','macs','compute_cycles',
        'elapsed_ps','elapsed_time_origin','command_to_final_drain_ps','final_drain_tail_ps',
        'gmac_s','gops','dma_read_bus_bytes','dma_write_bus_bytes','dma_bus_bandwidth_GB_s',
        'dram_physical_bytes','dram_bandwidth_GB_s','sram_read_bytes','sram_write_bytes',
        'aggregate_read_bytes','aggregate_write_bytes','onchip_moved_bytes',
        'sequencer_cycle_accounting_exact','dram_tck_ps','rtl_sha256','simulator_sha256',
        'testbench_sha256','dram_library_sha256','dram_config_sha256','trace_sha256',
        'output_sha256','time_kind','timing_validated')
    measurements=[]
    for prefix in ('functional','generated_sram_gemm','generated_sram_gemm8'):
        pattern='*/metrics.json'if prefix=='generated_sram_gemm8'else'check_*/metrics.json'
        for path in sorted((out/prefix).glob(pattern)):
            data=read(path)
            if not data.get('passed'):
                continue
            if (data.get('core_frequency_hz'),data.get('memory_frequency_hz'),data.get('dram_tck_ps'))!=(1e9,5e8,1000):
                raise ValueError('measurement uses another clock configuration: '+str(path))
            sample={key:data[key] for key in fields if key in data}
            sample.update(kind=prefix+'/'+path.parent.name,evidence=evidence(path),
                source_cohort_sha256=cohort['cohort_sha256'],physical_frequency_certified=False,
                macro_model='generated-native-pin-wrapper' if prefix!='functional' else 'frozen-logical-macro-interface')
            if prefix!='functional':
                sample['macro_package_manifest_sha256']=(native8['macro_package_manifest_sha256']
                    if prefix=='generated_sram_gemm8'else package['manifest_sha256'])
            measurements.append(sample)
    performance=dict(kind='configured_clock_projection',physical_frequency_certified=False,
        core_period_ps=1000,memory_period_ps=2000,dram_period_ps=1000,
        dma_per_direction_theoretical_GB_s=8.0,workloads=measurements,
        note='Includes arbitration, partial-word RMW, response waits and final write drain. Different array-size workloads are not a speedup comparison.')
    (out/'performance_projection.json').write_text(json.dumps(performance,indent=2)+'\n')
    print(json.dumps(dict(functional_steps=regular['steps'],generated_macro_steps=generated['steps'],
        functional_model_acceptance=matrix['functional_model_acceptance'],full_hardware_plan_complete=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/migration_asap7_f2')
    run(p.parse_args().out.resolve())
