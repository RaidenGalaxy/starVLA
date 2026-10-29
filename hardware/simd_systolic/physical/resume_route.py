#!/usr/bin/env python3
"""Resume detailed routing from a frozen, already repaired global-route DB."""
import argparse,fcntl,json,os,re,shutil,subprocess,sys
from pathlib import Path
from run import ROOT,sha,tool


def resume(source,label):
    source=source.resolve()
    if not re.fullmatch('[a-z][a-z0-9_]{0,47}',label):raise ValueError('invalid routing iteration label')
    checkpoint=source/'post_global_route.odb'
    identity=json.loads((source/'build_identity.json').read_text())
    if sha(source/'synth.v')!=identity['mapped_sha256']:raise ValueError('mapped source identity changed')
    out=source.parent/(source.name+'_'+label);out.mkdir(exist_ok=True)
    guard=(out/'.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if(out/'flow.tcl').exists():raise ValueError('routing iteration already exists; choose a new label')
    for name in ('inputs','generated'):
        if(source/name).exists():shutil.copytree(source/name,out/name)
    for name in ('synth.v','mapping.lib','cell_statistics.json','synth.log','design.sdc'):
        if(source/name).exists():shutil.copy2(source/name,out/name)
    shutil.copy2(checkpoint,out/'seed.odb')
    if sha(out/'seed.odb')!=sha(checkpoint):raise ValueError('routing DB changed during snapshot')
    identity.update(route_resume_parent=str(source),route_resume_checkpoint_sha256=sha(checkpoint),
        route_resume_identity_sha256=sha(source/'build_identity.json'),
        route_resume_log_sha256=sha(source/'route.log'))
    (out/'build_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    flow='''set_thread_count 8
read_lef $::env(SIMD_TECH_LEF)
read_lef $::env(SIMD_CELL_LEF)
read_liberty $::env(SIMD_LIB)
read_db $::env(SIMD_OUT)/seed.odb
read_sdc $::env(SIMD_OUT)/design.sdc
set_propagated_clock [all_clocks]
set_wire_rc -signal -layer M3
set_wire_rc -clock -layer M5
set_dont_use [get_lib_cells {*/*x1p*_ASAP7* */*xp*_ASAP7* */SDF* */ICG*}]
set_routing_layers -signal M2-M7 -clock M4-M7
set_global_routing_layer_adjustment M2-M3 0.5
set_global_routing_layer_adjustment M4-M7 0.25
'''
    flow+='detailed_route -output_drc'+(ROOT/'physical/run.tcl').read_text().split('detailed_route -output_drc',1)[1]
    (out/'flow.tcl').write_text(flow)
    env=os.environ.copy();local=ROOT/'.deps/tools/openroad';pdk=ROOT/'.deps/asap7'
    env['LD_LIBRARY_PATH']=str(local/'opt/or-tools/lib')+':'+str(local/'usr/lib/x86_64-linux-gnu')
    env.update(SIMD_OUT=str(out),SIMD_TECH_LEF=str(pdk/'lef/asap7_tech_1x_201209.lef'),
        SIMD_CELL_LEF=str(pdk/'lef/asap7sc7p5t_28_R_1x_220121a.lef'),
        SIMD_LIB=str(pdk/'asap7_RVT_TT.lib'),SIMD_RCX=str(pdk/'rcx_patterns.rules'))
    try:
        with(out/'route.log').open('w')as log:
            subprocess.run([tool('openroad'),'-exit',str(out/'flow.tcl')],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
        subprocess.run([sys.executable,str(ROOT/'physical/finalize.py'),str(out)],check=True,stdout=subprocess.DEVNULL,env=env)
        print((out/'timing.json').read_text())
    except Exception as error:
        (out/'failure.json').write_text(json.dumps(dict(status='resumed_route_failed',error=str(error)),indent=2)+'\n')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('--label',required=True)
    args=parser.parse_args();resume(args.source,args.label)
