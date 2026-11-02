#!/usr/bin/env python3
"""Exercise the real post-hold CLI guard with changed clock/ODB inputs."""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from run import ROOT,sha


def run(args):
    source=args.source.resolve();out=args.out.resolve()
    if out.exists():raise ValueError('preserve prior resume-guard results')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'check_driver.py')
    stage='post_detailed_route'if args.detailed_route else'post_global_route'if args.global_route else'post_initial_hold'
    stages=source/('detailed_route_checkpoint.json'if args.detailed_route else'global_route_checkpoint.json'if args.global_route else'initial_hold_checkpoint.json')
    original=json.loads(stages.read_text())
    for filename,digest in original['evidence'].items():
        if sha(Path(filename))!=digest:raise ValueError('actual source checkpoint changed: '+filename)
    checks=[]
    cases=('clock','database','netlist')if args.detailed_route else('clock','database','guides')if args.global_route else('clock','database')
    for case in cases:
        with tempfile.TemporaryDirectory(prefix='resume_guard_',dir=ROOT/'.build')as directory:
            candidate=Path(directory)
            for name in('inputs','build_identity.json','cts_timing.json','synth.v',
                    stages.name,'route_parent.json'):
                (candidate/name).symlink_to(source/name,target_is_directory=(source/name).is_dir())
            if case=='clock':
                text=(source/'design.sdc').read_text()
                if text.count('-period 1.000000')!=1:raise ValueError('source clock is not the expected1ns')
                (candidate/'design.sdc').write_text(text.replace('-period 1.000000','-period 2.000000'))
                (candidate/(stage+'.odb')).symlink_to(source/(stage+'.odb'))
                modified='design.sdc'
            elif case=='database':
                (candidate/'design.sdc').symlink_to(source/'design.sdc')
                (candidate/(stage+'.odb')).write_text('deliberately invalid database\n')
                modified=stage+'.odb'
            elif case=='guides':
                (candidate/'design.sdc').symlink_to(source/'design.sdc')
                (candidate/(stage+'.odb')).symlink_to(source/(stage+'.odb'))
                (candidate/'post_global_route.guide').write_text('deliberately invalid route guides\n')
                modified='post_global_route.guide'
            else:
                (candidate/'design.sdc').symlink_to(source/'design.sdc')
                (candidate/(stage+'.odb')).symlink_to(source/(stage+'.odb'))
                (candidate/'post_detailed_route.v').write_text('deliberately invalid routed netlist\n')
                modified='post_detailed_route.v'
            if args.global_route and case!='guides':
                (candidate/'post_global_route.guide').symlink_to(source/'post_global_route.guide')
            if args.detailed_route and case!='netlist':
                (candidate/'post_detailed_route.v').symlink_to(source/'post_detailed_route.v')
            label='rejected_resume_'+candidate.name
            output=ROOT/('.build/physical/asap7_TT_int64_2_1x1_1ns_'+label)
            command=[sys.executable,str(ROOT/'physical/route_arithmetic.py'),'--source',str(candidate),
                '--label',label,'--resume-detailed-route'if args.detailed_route else'--resume-global-route'if args.global_route else'--resume-initial-hold','--timeout','30']
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=90)
            log=out/(case+'.log');log.write_text(result.stdout+result.stderr)
            expected='post-hold evidence does not bind the actual copied '+modified
            if result.returncode==0 or expected not in result.stderr or output.exists():
                raise AssertionError('changed '+case+' was not rejected before starting implementation: '+result.stderr)
            changed=out/('changed_'+modified);shutil.copy2(candidate/modified,changed)
            checks.append(dict(case=case,returncode=result.returncode,diagnostic=expected,
                rejected_before_output_creation=True,changed_input_sha256=sha(changed),log_sha256=sha(log)))
    evidence=[stages,ROOT/'physical/route_arithmetic.py',ROOT/'physical/run.py',ROOT/'physical/run.tcl',
        *[p for p in out.iterdir()if p.is_file()]]
    record=dict(kind='actual_cli_post_hold_checkpoint_rejection_check',passed=True,checks=checks,
        resumed_stage=stage,
        source_checkpoint=str(stages),core_target_ns=1,arithmetic_stages=2,physical_frequency_certified=False,
        evidence={str(p):sha(p)for p in evidence})
    (out/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items()if k!='evidence'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--global-route',action='store_true')
    p.add_argument('--detailed-route',action='store_true');args=p.parse_args()
    if args.global_route and args.detailed_route:p.error('choose one saved stage')
    run(args)
