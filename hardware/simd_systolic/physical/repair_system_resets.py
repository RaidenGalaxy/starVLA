#!/usr/bin/env python3
"""Repair authenticated high-fanout reset or control nets with the pinned engine.

Pure buffer/inverter roots, clock/data connections and protected masters are
checked. This placement-STA diagnostic does not replace routed signoff.
"""
import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from asap7_platform import profile
from prepare_xyce import sha
from sram_package import load_package
from simd_eval.architecture import ARCH


def run(args):
    source=args.source.resolve();proof_path=source/'verification.json'
    prior=json.loads(proof_path.read_text())
    from async_mapping_policy import validate_checkpoint
    validate_checkpoint(source,prior)
    checkpoint=source/('repaired.odb'if args.controls or args.analyze_only else'placed.odb')
    if prior['core_period_ns']!=1 or prior['memory_period_ns']!=2:
        raise ValueError('reset diagnostic cannot change the target clocks')
    for path,digest in prior['evidence'].items():
        if sha(path)!=digest:raise ValueError('source placement evidence changed: '+path)
    if prior['evidence'].get(str(checkpoint))!=sha(checkpoint):raise ValueError('placed checkpoint missing or changed')
    if args.controls or args.analyze_only:
        if not prior['logical_roots_and_protected_data_clock_masters_preserved']:
            raise ValueError('control repair needs a completed authenticated reset repair')
        cts_source=source;cts_record=prior;visited=set()
        while 'SIMD_PROTECTED_CTS_CLOCKS_AND_SIGNALS_VERIFIED'not in(cts_source/'openroad.log').read_text():
            # Successive bounded ECOs retain and authenticate every parent.
            # A small fixed depth rejected valid continued repair sequences.
            if cts_source in visited or len(visited)>=128:raise ValueError('invalid CTS checkpoint ancestry')
            visited.add(cts_source)
            cts_proof=Path(cts_record['source_verification'])
            if sha(cts_proof)!=cts_record['source_verification_sha256']:
                raise ValueError('control/reset CTS parent changed')
            cts_source=cts_proof.parent;cts_record=json.loads(cts_proof.read_text())
        if cts_record['evidence'].get(str(cts_source/'openroad.log'))!=sha(cts_source/'openroad.log'):
            raise ValueError('source CTS audit log changed')
    else:cts_source=source
    if 'SIMD_PROTECTED_CTS_CLOCKS_AND_SIGNALS_VERIFIED'not in(cts_source/'openroad.log').read_text():
        raise ValueError('source did not complete protected dual CTS')
    macro=load_package(args.sram_package,ARCH);technology=profile('TT')
    if macro['manifest_sha256']!=prior['physical_macro_package_manifest_sha256']:
        raise ValueError('physical SRAM revision differs from the checkpoint')
    if prior['evidence'].get(str(technology['liberty']))!=sha(technology['liberty']):
        raise ValueError('source standard-cell library differs')
    out=args.out.resolve()
    if out.exists():raise ValueError('preserve prior reset diagnostics')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'experiment_driver.py')
    for name in('cdc_constraints.tcl','protected_power.tcl'):shutil.copy2(source/name,out/name)
    shutil.copy2(ROOT/'physical/reset_repair.tcl',out/'reset_repair.tcl')
    (out/'design.sdc').write_text((source/'design.sdc').read_text().replace(str(source),str(out)))
    if args.analyze_only:
        repair='puts SIMD_SYSTEM_CHECKPOINT_LOGIC_REUSED\n'
        reset_reports=''
    elif args.controls:
        shutil.copy2(ROOT/'physical/control_repair.tcl',out/'control_repair.tcl')
        from netlist_fanout import census as port_census
        netlist=source/'repaired.v'
        if prior['evidence'].get(str(netlist))!=sha(netlist):raise ValueError('source netlist changed')
        shutil.copy2(ROOT/'physical/netlist_fanout.py',out/'fanout_census_driver.py')
        census,clock_nets=port_census(netlist)
        explicit=list(dict.fromkeys(args.control_net))
        if explicit:
            for name in explicit:
                if name not in census or name in clock_nets:
                    raise ValueError('explicit control is absent from signal census: '+name)
            candidates=[(name,census[name])for name in explicit]
        else:
            candidates=[(name,count)for name,count in census.most_common()if count>=args.minimum and name not in clock_nets][:args.limit*8]
        (out/'fanout_census.json').write_text(json.dumps(dict(source_netlist=str(netlist),source_sha256=sha(netlist),
            scope='scalar cell port references; actual OpenDB input counts checked before repair',
            selection='explicit authenticated net names'if explicit else'largest scalar signal port count',
            candidates=candidates),indent=2)+'\n')
        if not candidates:raise ValueError('no control candidates in the actual netlist')
        names=' '.join('{'+name+'}'for name,_ in candidates)
        repair=f'''source {{{out/'control_repair.tcl'}}}
set targets [simd_repair_control_fanout {{{names}}} {args.minimum} {args.limit} 50 {{{out/'control_nets.tsv'}}} {{{out/'control_checkpoint.odb'}}}]
puts "SIMD_CONTROL_LOGIC_PRESERVED [llength $targets]"
'''
        reset_reports=''
    else:
        repair=f'''set targets [simd_repair_reset_fanout {args.minimum} 50 {{{out/'reset_nets.tsv'}}}]
puts "SIMD_RESET_LOGIC_PRESERVED [llength $targets]"
'''
        reset_reports=f'''report_checks -path_delay max -to [get_pins $targets] -group_count 10 -format full_clock_expanded -digits 9 > {{{out/'reset_setup.rpt'}}}
report_checks -path_delay min -to [get_pins $targets] -group_count 10 -format full_clock_expanded -digits 9 > {{{out/'reset_hold.rpt'}}}
'''
    update=''if args.analyze_only else f'''if {{![llength $targets]}} {{error "No high-fanout signal network selected"}}
detailed_placement
global_connect
source {{{out/'protected_power.tcl'}}}
simd_connect_protected_supplies
check_placement -verbose
write_db {{{out/'repaired.odb'}}}
write_verilog {{{out/'repaired.v'}}}
estimate_parasitics -placement
'''
    electrical=''if args.analyze_only else f'''report_check_types -max_slew -max_capacitance -violators > {{{out/'electrical.rpt'}}}'''
    script=out/'repair.tcl';script.write_text(f'''set_thread_count 8
read_liberty {{{technology['liberty']}}}
read_liberty {{{macro['liberty']}}}
read_db {{{checkpoint}}}
set_cmd_units -time ns -capacitance fF
read_sdc {{{out/'design.sdc'}}}
set_propagated_clock [all_clocks]
source {{{technology['root']/'setRC.tcl'}}}
set_wire_rc -signal -layer M3
set_wire_rc -clock -layer M5
set_dont_use [get_lib_cells {{*/*x1p*_ASAP7* */*xp*_ASAP7* */SDF* */ICG*}}]
estimate_parasitics -placement
source {{{out/'reset_repair.tcl'}}}
{repair}
{update}
report_clock_properties [all_clocks] > {{{out/'clocks.rpt'}}}
puts SIMD_RESET_REPAIRED_SETUP
report_worst_slack -max -digits 9
puts SIMD_RESET_REPAIRED_HOLD
report_worst_slack -min -digits 9
{reset_reports}
report_checks -path_delay max -group_count 3 -format full_clock_expanded -digits 9 > {{{out/'setup.rpt'}}}
{electrical}
puts SIMD_RESET_REPAIR_DIAGNOSTIC_COMPLETE
''')
    local=ROOT/'.deps/tools/openroad';env=os.environ.copy()
    env['LD_LIBRARY_PATH']=':'.join([str(local/'opt/or-tools/lib'),str(local/'usr/lib/x86_64-linux-gnu'),env.get('LD_LIBRARY_PATH','')])
    log=out/'openroad.log';timed_out=False
    with log.open('w')as stream:
        process=subprocess.Popen([str(local/'usr/bin/openroad'),'-exit',str(script)],env=env,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        (out/'running.json').write_text(json.dumps(dict(pid=process.pid,timeout_seconds=args.timeout))+'\n')
        try:code=process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            code=process.wait()
    text=log.read_text();timing={}
    for marker,kind in(('SETUP','setup'),('HOLD','hold')):
        match=re.search('SIMD_RESET_REPAIRED_'+marker+r'.*?worst slack\s+([-+\d.]+)',text,re.S)
        if match:timing[kind+'_wns_ns']=float(match[1])
    label='CONTROL'if args.controls else'RESET'
    count=re.search('SIMD_'+label+r'_NETS_REPAIRED (\d+)',text)
    selected=int(count[1])if count else 0
    passed=code==0 and not timed_out and(args.analyze_only or selected>0)and 'SIMD_RESET_REPAIR_DIAGNOSTIC_COMPLETE'in text
    result=dict(kind='placed_system_sta_diagnostic'if args.analyze_only else'focused_system_control_tree_placement_diagnostic'if args.controls else'focused_system_reset_tree_placement_diagnostic',passed=passed,returncode=code,timed_out=timed_out,
        timeout_seconds=args.timeout,source_verification=str(proof_path),source_verification_sha256=sha(proof_path),
        root_cts_verification=str(cts_source/'verification.json'),root_cts_verification_sha256=sha(cts_source/'verification.json'),
        source_rtl_sha256=prior['source_rtl_sha256'],source_mapped_sha256=prior['source_mapped_sha256'],
        physical_macro_package_manifest_sha256=macro['manifest_sha256'],core_period_ns=1,memory_period_ns=2,
        reset_nets_repaired=0 if args.controls else selected,control_nets_repaired=selected if args.controls else 0,
        individually_guarded_control_nets=re.findall(r'(?m)^SIMD_CONTROL_REPAIR_END (\S+)',text),
        connectivity_checkpoint_completed_nets=re.findall(r'(?m)^SIMD_CONTROL_CHECKPOINT_WRITTEN (\S+)',text),
        connectivity_checkpoint_is_legalized=False,
        minimum_total_net_input_sinks=args.minimum,
        explicit_control_nets=args.control_net,
        logical_roots_and_protected_data_clock_masters_preserved=(prior['logical_roots_and_protected_data_clock_masters_preserved']and 'SIMD_SYSTEM_CHECKPOINT_LOGIC_REUSED'in text)if args.analyze_only else'SIMD_'+label+'_LOGIC_PRESERVED'in text,
        reused_immutable_checkpoint_only=args.analyze_only,electrical_report_generated=not args.analyze_only,
        per_net_engine='Pinned OpenROAD rsz::repair_net_cmd; independently exercised on a small actual design',
        timing=timing,signal_routing_complete=False,extracted_sta=False,
        whole_system_power_grid_rechecked=False,physical_frequency_certified=False,
        **{key:prior[key]for key in('qualified_async_mapping_root','async_mapping_correction_verified',
            'async_correction_verification','async_correction_verification_sha256')if key in prior},
        evidence={str(p):sha(p)for p in [proof_path,checkpoint,technology['liberty'],macro['liberty'],
            *[p for p in out.iterdir()if p.is_file()]]})
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='evidence'},indent=2))
    if not passed:raise RuntimeError('reset repair diagnostic failed or timed out; evidence retained')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--sram-package',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--minimum',type=int,default=256)
    p.add_argument('--controls',action='store_true',help='repair the largest signal nets after a completed reset diagnostic')
    p.add_argument('--control-net',action='append',default=[],help='select an explicit actual signal net; repeat for critical-path diagnostics')
    p.add_argument('--analyze-only',action='store_true',help='time a checked saved database without changing placement or connectivity')
    p.add_argument('--limit',type=int,default=16,help='maximum control nets to repair in this bounded pass')
    p.add_argument('--timeout',type=int,default=1800);args=p.parse_args()
    if args.minimum<2 or not 1<=args.timeout<=3600:p.error('minimum>=2 and timeout1..3600 required')
    if not 1<=args.limit<=64:p.error('control net limit must be within1..64')
    if args.controls and args.analyze_only:p.error('choose control repair or immutable timing analysis')
    if args.control_net and not args.controls:p.error('explicit net selection requires --controls')
    run(args)
