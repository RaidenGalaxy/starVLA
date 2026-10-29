#!/usr/bin/env python3
"""Reroute authenticated repaired placement with a fresh global router."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from run import ROOT,sha,tool,summarize
from export_macro import tcl
from simd_eval.constraints import core_policy


def reroute(source,label,allow_congestion=False,threads=4):
    source=source.resolve()
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,47}',label):raise ValueError('invalid reroute label')
    checkpoint=json.loads((source/'verification_repair_checkpoint.json').read_text())
    if not checkpoint.get('passed')or checkpoint.get('scope')!='unrouted_repair_checkpoint':
        raise ValueError('an authenticated repair checkpoint is required')
    for name,path in checkpoint['files'].items():
        if sha(path)!=checkpoint['artifacts_sha256'][name]:raise ValueError('repair checkpoint changed: '+name)
    identity=json.loads((source/'build_identity.json').read_text())
    if identity['rtl_sha256']!=checkpoint['rtl_sha256']:raise ValueError('checkpoint source identity differs')
    for name,digest in identity['input_manifest'].items():
        if sha(source/'inputs'/name)!=digest:raise ValueError('source RTL snapshot changed: '+name)
    period=checkpoint['period_ns'];sdc=source/'design.sdc'
    if not core_policy(sdc.read_text(),period)['clean']:raise ValueError('complete core constraints required')
    pdk=ROOT/'.deps/asap7';lib=pdk/'asap7_RVT_TT.lib'
    if sha(lib)!=identity['liberty_sha256']:raise ValueError('process library changed')
    out=source.parent/(source.name+'_'+label);out.mkdir(exist_ok=True)
    guard=(out/'.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if(out/'flow.tcl').exists():raise ValueError('reroute iteration exists; choose another label')
    for name in('inputs','generated'):
        if(source/name).exists():shutil.copytree(source/name,out/name,dirs_exist_ok=True)
    for name in('synth.v','mapping.lib','cell_statistics.json','synthesis.json','design.sdc',
                'NangateSyncControls.v','sequential_mapping_proof.json'):
        if(source/name).exists():shutil.copy2(source/name,out/name)
    shutil.copy2(source/'post_repair.odb',out/'seed_repair.odb')
    shutil.copy2(source/'post_repair.v',out/'repair_netlist.v')
    if sha(out/'seed_repair.odb')!=checkpoint['artifacts_sha256']['post_repair.odb']:
        raise ValueError('repair database changed during copying')
    identity.update(repair_checkpoint_from=str(source),repair_checkpoint_report_sha256=sha(source/'verification_repair_checkpoint.json'),
        seed_checkpoint_sha256=sha(out/'seed_repair.odb'),repair_netlist_sha256=sha(out/'repair_netlist.v'),
        global_router_action='fresh_after_extracted_timing_repair',global_route_allow_congestion=allow_congestion,
        physical_worker_threads=threads,analysis_period_ns=period)
    (out/'build_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    tail=(ROOT/'physical/run.tcl').read_text().split('detailed_route -output_drc',1)[1]
    flow=f'''set_thread_count {threads}
read_lef {tcl(pdk/'lef/asap7_tech_1x_201209.lef')}
read_lef {tcl(pdk/'lef/asap7sc7p5t_28_R_1x_220121a.lef')}
read_liberty {tcl(lib)}
read_db {tcl(out/'seed_repair.odb')}
read_sdc {tcl(out/'design.sdc')}
set_propagated_clock [all_clocks]
set_wire_rc -signal -layer M3
set_wire_rc -clock -layer M5
set_dont_use [get_lib_cells {{*/*x1p*_ASAP7* */*xp*_ASAP7* */SDF* */ICG*}}]
set_routing_layers -signal M2-M7 -clock M4-M7
set_global_routing_layer_adjustment M2-M3 0.5
set_global_routing_layer_adjustment M4-M7 0.25
write_verilog {tcl(out/'loaded_checkpoint.v')}
puts "SIMD_FRESH_ROUTER_START"
global_route -guide_file {tcl(out/'route.guide')} -congestion_iterations 100{' -allow_congestion'if allow_congestion else''}
estimate_parasitics -global_routing
write_db {tcl(out/'post_global_route.odb')}
report_checks -path_delay max -group_count 20 -format full_clock_expanded -digits 9 > {tcl(out/'post_global_route_setup.rpt')}
report_checks -path_delay min -group_count 20 -format full_clock_expanded -digits 9 > {tcl(out/'post_global_route_hold.rpt')}
detailed_route -output_drc'''+tail
    (out/'flow.tcl').write_text(flow)
    env=os.environ.copy();local=ROOT/'.deps/tools/openroad'
    env['LD_LIBRARY_PATH']=str(local/'opt/or-tools/lib')+':'+str(local/'usr/lib/x86_64-linux-gnu')
    env.update(SIMD_OUT=str(out),SIMD_RCX=str(pdk/'rcx_patterns.rules'))
    try:
        with(out/'route.log').open('w')as log:
            subprocess.run([tool('openroad'),'-exit',str(out/'flow.tcl')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if sha(out/'loaded_checkpoint.v')!=identity['repair_netlist_sha256']:
            raise ValueError('database connectivity differs from the authenticated repair checkpoint')
        report=summarize(out,identity,period);print(json.dumps(report,indent=2))
    except Exception as error:
        (out/'failure.json').write_text(json.dumps(dict(status='fresh_checkpoint_reroute_failed',error=str(error),log=str(out/'route.log')),indent=2)+'\n')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('--label',required=True)
    parser.add_argument('--allow-global-congestion',action='store_true',help='permit global estimates with overflow; final detailed-route DRC and timing still must close')
    parser.add_argument('--threads',type=int,choices=[1,2,4,8],default=4)
    args=parser.parse_args();reroute(args.source,args.label,args.allow_global_congestion,args.threads)
