#!/usr/bin/env python3
"""Repair extracted timing in a new, authenticated physical iteration."""
import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from run import ROOT, sha, tool, summarize


def repair(source,label,period=None,hold_margin=None,setup_margin=None,repair_only=False,allow_initial_global_congestion=False):
    source=source.resolve()
    if not label or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in label):
        raise ValueError('use a lowercase alphanumeric ECO label')
    proof=json.loads((source/'timing.json').read_text())
    period=proof['period_ns']if period is None else period
    if period<=0.15:raise ValueError('invalid analysis period')
    margin=(.08 if proof['scope']=='fp64'else .05)if hold_margin is None else hold_margin
    if not 0<=margin<=.5:raise ValueError('hold margin must be in [0,0.5] ns')
    setup=proof.get('setup_repair_margin_ns',.05)if setup_margin is None else setup_margin
    if not 0<=setup<=.5:raise ValueError('setup margin must be in [0,0.5] ns')
    for name,path in proof['evidence'].items():
        if sha(path)!=proof['evidence_sha256'][name]:raise RuntimeError('changed source evidence: '+name)
    out=source.parent/(source.name+'_'+label)
    out.mkdir(exist_ok=True)
    guard=(out/'.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if(out/'flow.tcl').exists():raise RuntimeError('ECO iteration exists; choose a new label')
    for name in('inputs','generated'):
        shutil.copytree(source/name,out/name,dirs_exist_ok=True)if(source/name).exists()else None
    for name in('synth.v','mapping.lib','design.sdc','cell_statistics.json','synthesis.json',
                'NangateSyncControls.v','sequential_mapping_proof.json'):
        if(source/name).exists():shutil.copy2(source/name,out/name)
    for src,dst in [('design.odb','seed.odb'),('design.spef','seed.spef')]:
        shutil.copy2(source/src,out/dst)
    shutil.copy2(source/'timing.json',out/'seed_timing.json')
    for name in ('setup.rpt','hold.rpt','coverage.rpt','coverage_analysis.json','electrical.rpt','route.drc'):
        shutil.copy2(source/name,out/('seed_'+name))
    identity=json.loads((source/'build_identity.json').read_text())
    identity.update(eco_parent=str(source),eco_parent_timing_sha256=sha(source/'timing.json'),
                    seed_database_sha256=sha(out/'seed.odb'),seed_spef_sha256=sha(out/'seed.spef'),
                    analysis_period_ns=period,routed_hold_repair_margin_ns=margin,setup_repair_margin_ns=setup,
                    optimization_parasitics='extracted_seed_with_incremental_global_route_updates_on_modified_nets',
                    optimization_reference='https://github.com/The-OpenROAD-Project/OpenROAD/blob/a008522d8/src/rsz/src/EstimateWireParasitics.cc')
    identity['initial_global_route_allow_congestion']=allow_initial_global_congestion
    if period!=proof['period_ns']:
        sdc=out/'design.sdc';original=sdc.read_text()
        if '-waveform'in original:raise ValueError('explicit clock waveform needs manual review before period change')
        updated,count=re.subn(r'(?m)^(create_clock[^\n]*?-period\s+)[\d.]+',
            lambda match:match[1]+f'{period:g}',original)
        if count!=1:raise ValueError('expected one core clock in the extracted design')
        (out/'seed_constraints.sdc').write_text(original);sdc.write_text(updated)
        identity['constraint_change']=f'analysis period changed from {proof["period_ns"]:g} to {period:g} ns; no timing exceptions added'
    (out/'build_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    flow=(ROOT/'physical/eco.tcl').read_text()
    # Share extraction, output files and the strict signoff reports with the
    # normal flow. The pre-existing routed result remains untouched.
    flow+='detailed_route -output_drc'+(ROOT/'physical/run.tcl').read_text().split('detailed_route -output_drc',1)[1]
    (out/'flow.tcl').write_text(flow)
    env=os.environ.copy();local=ROOT/'.deps/tools/openroad';pdk=ROOT/'.deps/asap7'
    env['LD_LIBRARY_PATH']=str(local/'opt/or-tools/lib')+':'+str(local/'usr/lib/x86_64-linux-gnu')
    env.update(SIMD_OUT=str(out),SIMD_TECH_LEF=str(pdk/'lef/asap7_tech_1x_201209.lef'),
        SIMD_CELL_LEF=str(pdk/'lef/asap7sc7p5t_28_R_1x_220121a.lef'),
        SIMD_LIB=str(pdk/'asap7_RVT_TT.lib'),SIMD_RCX=str(pdk/'rcx_patterns.rules'),
        SIMD_HOLD_MARGIN=str(margin),SIMD_SETUP_MARGIN=str(setup))
    if repair_only:env['SIMD_ECO_REPAIR_ONLY']='1'
    if allow_initial_global_congestion:env['SIMD_ECO_ALLOW_INITIAL_CONGESTION']='1'
    try:
        with(out/'route.log').open('w')as log:
            subprocess.run([tool('openroad'),'-exit',str(out/'flow.tcl')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        if repair_only:
            files=('post_repair.odb','post_repair.v','post_repair_mixed.spef','post_repair_setup.rpt','post_repair_hold.rpt')
            if 'SIMD_ECO_REPAIR_CHECKPOINT'not in(out/'route.log').read_text():
                raise RuntimeError('repair checkpoint was not completed')
            report=dict(passed=True,scope='unrouted_repair_checkpoint',source_scope=identity['scope'],
                stages=identity['stages'],rows=identity['rows'],cols=identity['cols'],
                rtl_sha256=identity['rtl_sha256'],period_ns=period,
                timing_status='unrouted_repair_checkpoint',timing_validated=False,
                parasitics_model=identity['optimization_parasitics'],
                source_directory=str(source),source_timing_sha256=sha(source/'timing.json'),
                files={name:str(out/name)for name in files},
                artifacts_sha256={name:sha(out/name)for name in files},
                limitation='Repaired placement only; global/detailed routing, new extraction and routed numerical validation remain required.')
            (out/'verification_repair_checkpoint.json').write_text(json.dumps(report,indent=2)+'\n')
        else:report=summarize(out,identity,period)
        print(json.dumps(report,indent=2))
    except Exception as error:
        (out/'failure.json').write_text(json.dumps(dict(status='eco_failed',error=str(error),log=str(out/'route.log')),indent=2)+'\n')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--label',required=True)
    p.add_argument('--period',type=float,help='explicit new analysis period in ns; never treated as the original frequency')
    p.add_argument('--hold-margin',type=float)
    p.add_argument('--setup-margin',type=float,help='optimization guard band only; final constraints remain strict')
    p.add_argument('--repair-only',action='store_true',help='save repaired placement and mixed estimates before rerouting; does not certify timing')
    p.add_argument('--allow-initial-global-congestion',action='store_true',help='allow overflow in optimization guides; final detailed-route DRC and STA still must pass')
    args=p.parse_args();repair(args.source,args.label,args.period,args.hold_margin,args.setup_margin,args.repair_only,args.allow_initial_global_congestion)
