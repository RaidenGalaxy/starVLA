#!/usr/bin/env python3
"""Route a verified F2 component CTS checkpoint in a separate bounded run."""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from run import ARCH,ROOT,sha


def run(args):
    source=args.source.resolve();path=source/'cts_timing.json'
    cts=json.loads(path.read_text());identity=json.loads((source/'build_identity.json').read_text())
    target=identity['scope']
    resumed_stage='post_detailed_route'if args.resume_detailed_route else'post_global_route'if args.resume_global_route else'post_initial_hold'if args.resume_initial_hold else'post_cts'
    resume_saved=resumed_stage!='post_cts'
    if target not in('fp64','int64')or cts['stages']!=2 or cts['period_ns']!=1:
        raise ValueError('requires F2 FP64/I64 CTS at the unchanged1ns target')
    if cts['architecture_sha256']!=ARCH['architecture_sha256']or cts['process_corner']!='TT':
        raise ValueError('component architecture or process corner differs')
    if sha(Path(identity['liberty_file']))!=identity['liberty_sha256']:
        raise ValueError('component standard-cell library changed')
    for name,key in [('synth.v','mapped_sha256')]+([]if resume_saved else[('post_cts.odb','checkpoint_sha256')]):
        if sha(source/name)!=cts[key]:raise ValueError('source CTS input changed: '+name)
    for name,digest in identity['input_manifest'].items():
        if sha(source/'inputs'/name)!=digest:raise ValueError('source RTL changed: '+name)
    proof=source/('check_mapped/verification.json'if target=='int64'else'gate_verification.json')
    if resume_saved:
        stage_path=source/('detailed_route_checkpoint.json'if args.resume_detailed_route else'global_route_checkpoint.json'if args.resume_global_route else'initial_hold_checkpoint.json');stage=json.loads(stage_path.read_text())
        for filename,digest in stage['evidence'].items():
            if sha(Path(filename))!=digest:raise ValueError('post-hold source evidence changed: '+filename)
        required=[resumed_stage+'.odb','design.sdc']+(['post_global_route.guide']if args.resume_global_route else['post_detailed_route.v']if args.resume_detailed_route else[])
        for name in required:
            expected=[digest for filename,digest in stage['evidence'].items()if Path(filename).name==name]
            if expected!=[sha(source/name)]:raise ValueError('post-hold evidence does not bind the actual copied '+name)
        if stage['rtl_sha256']!=identity['rtl_sha256']or stage['arithmetic_stages']!=2 or stage['core_period_ns']!=1:
            raise ValueError('post-hold source changes stages/period/RTL')
        original=json.loads((source/'route_parent.json').read_text())
        proof=Path(original['source_mapped_numerical'])
        if sha(proof)!=original['source_mapped_numerical_sha256']:raise ValueError('mapped parent proof changed')
    numerical=json.loads(proof.read_text())
    if not numerical['passed']or numerical['rtl_sha256']!=cts['rtl_sha256']or numerical['netlist_sha256']!=cts['mapped_sha256']:
        raise ValueError('requires matching mapped numerical verification')
    out=ROOT/f'.build/physical/asap7_TT_{target}_2_1x1_1ns_{args.label}'
    if out.exists():raise ValueError('preserve prior physical runs; choose a new label')
    shutil.copytree(source,out,ignore=shutil.ignore_patterns('check_*','.lock','attempt_*',
        'route_execution.json','running.json','execution.log','timing.json','failure.json'))
    shutil.copy2(__file__,out/'route_experiment_driver.py')
    parent=dict(source=str(source),source_cts_sha256=sha(path),source_identity_sha256=sha(source/'build_identity.json'),
        source_mapped_numerical=str(proof),source_mapped_numerical_sha256=sha(proof),
        checkpoint_sha256=sha(source/(resumed_stage+'.odb'))if resume_saved else cts['checkpoint_sha256'],
        resumed_stage=resumed_stage,
        mapped_sha256=cts['mapped_sha256'],rtl_sha256=cts['rtl_sha256'],
        arithmetic_stages=2,core_period_ns=1,physical_frequency_certified=False)
    (out/'route_parent.json').write_text(json.dumps(parent,indent=2)+'\n')
    if resume_saved:
        resume=dict(stage=resumed_stage,rtl_sha256=identity['rtl_sha256'],mapped_sha256=identity['mapped_sha256'],
            checkpoint_sha256=sha(out/(resumed_stage+'.odb')),sdc_sha256=sha(out/'design.sdc'),
            parent_manifest=str(stage_path),parent_manifest_sha256=sha(stage_path))
        if args.resume_global_route:resume['guides_sha256']=sha(out/'post_global_route.guide')
        if args.resume_detailed_route:resume['routed_netlist_sha256']=sha(out/'post_detailed_route.v')
        (out/'resume_checkpoint_manifest.json').write_text(json.dumps(resume,indent=2)+'\n')
    command=[sys.executable,str(ROOT/'physical/run.py'),'--target',target,'--stages','2','--rows','1',
        '--period','1','--label',args.label,'--reuse-mapped',
        '--resume-detailed-route'if args.resume_detailed_route else'--resume-global-route'if args.resume_global_route else'--resume-initial-hold'if args.resume_initial_hold else'--resume-cts']
    if args.detailed_route_only:command.append('--detailed-route-only')
    log=out/'execution.log';timed_out=False
    with log.open('w')as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
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
    report=dict(**parent,returncode=code,timed_out=timed_out,timeout_seconds=args.timeout,
        routed_sta_completed=code==0 and(out/'timing.json').is_file(),
        detailed_route_only=args.detailed_route_only,
        detailed_route_checkpoint_sha256=sha(out/'detailed_route_checkpoint.json')if args.detailed_route_only and(out/'detailed_route_checkpoint.json').exists()else None,
        timing_sha256=sha(out/'timing.json')if(out/'timing.json').is_file()else None,
        execution_log_sha256=sha(log),route_log_sha256=sha(out/'route.log')if(out/'route.log').is_file()else None)
    (out/'route_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if code or timed_out:raise RuntimeError('component route failed or timed out; evidence retained')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--label',required=True);p.add_argument('--resume-initial-hold',action='store_true')
    p.add_argument('--resume-global-route',action='store_true')
    p.add_argument('--resume-detailed-route',action='store_true');p.add_argument('--detailed-route-only',action='store_true')
    p.add_argument('--timeout',type=int,default=1800);args=p.parse_args()
    if not re.fullmatch('[a-z0-9_]+',args.label):p.error('label must contain only lowercase letters,digits,underscore')
    if sum((args.resume_initial_hold,args.resume_global_route,args.resume_detailed_route))>1:p.error('choose one saved stage')
    if args.resume_detailed_route and args.detailed_route_only:p.error('detailed route is already complete')
    if not 1<=args.timeout<=7200:p.error('timeout must be within1..7200seconds')
    run(args)
