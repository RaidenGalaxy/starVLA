#!/usr/bin/env python3
"""Verify sustained II=1 with sixteen independent INT64/FP64 contexts."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from runtime import validated_build
from simd_eval.reference import Format,Op,operate,to_bits


def check(stages,build_dir=None):
    dest=Path(build_dir)if build_dir else ROOT/f'.build/pe_{stages}_1x1';guard,caps=validated_build(dest)
    if caps['stages']!=stages:raise ValueError('build pipeline differs from requested check')
    trace=dest/'ii_vectors.txt';commands=[];expected={};tag=0
    for fmt in(Format.I64,Format.F64):
        acc=[0]*16
        for iteration in range(4):
            for ctx in range(16):
                a,b=(1,2)if fmt==Format.I64 else(to_bits(1,fmt),to_bits(2,fmt))
                acc[ctx],flags,overflow=operate(a,b,acc[ctx],fmt,Op.MAC,1)
                expected[tag]=(acc[ctx],flags,overflow)
                commands.append(f'{int(fmt)} 1 {a:x} {b:x} 1 {ctx} {int(iteration==0)} {tag}\n');tag+=1
    trace.write_text(''.join(commands))
    proc=subprocess.run([str(dest/'sim'),str(trace),'--no-stall'],capture_output=True,text=True,check=True,timeout=60)
    seen=set();issues=[];retirements=[]
    for line in proc.stdout.splitlines():
        if not line.startswith('O '):continue
        _,tag,data,flags,overflow,start,end=line.split();tag=int(tag)
        if tag in seen or(int(data,16),int(flags),int(overflow))!=expected[tag]:raise AssertionError('II stream numeric/tag mismatch')
        seen.add(tag);issues.append(int(start));retirements.append(int(end))
    if len(seen)!=len(expected)or any(b-a!=1 for a,b in zip(issues,issues[1:])):
        raise AssertionError('independent contexts did not sustain one issue per clock')
    if any(b-a!=1 for a,b in zip(retirements,retirements[1:])):raise AssertionError('result stream contains unexpected bubbles')
    latencies={end-start for start,end in zip(issues,retirements)}
    expected_latency=stages+caps.get('pe_state_read_latency',0)+caps.get('pe_commit_latency',0)
    if latencies!={expected_latency}:raise AssertionError('reported PE latency differs from actual response timing')
    if caps['feedback_issue_interval']<expected_latency+1:raise AssertionError('feedback interval allows reuse before the state write')
    report=dict(passed=True,stages=stages,contexts=16,vectors=len(seen),formats=['I64','F64'],
        issue_interval_cycles=1,retirement_interval_cycles=1,response_latency_cycles=expected_latency,
        feedback_issue_interval=caps['feedback_issue_interval'],
        rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'])
    (dest/'verification_ii.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stages',type=int,default=2);p.add_argument('--build-dir')
    args=p.parse_args();check(args.stages,args.build_dir)
