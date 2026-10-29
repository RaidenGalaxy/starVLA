#!/usr/bin/env python3
"""Exercise Kmax, exact INT64 accumulation and an independent wave drain model."""
import argparse
import json
import subprocess
from pathlib import Path
from runtime import validated_build
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/'python'))
from simd_eval.cycle_model import mesh_trace_cycles,verify_mesh_cycles


def check(stages,k,timeout=None,build_dir=None,format='I64',checkpoints=None):
    dest=Path(build_dir)if build_dir else ROOT/f'.build/array_{stages}_8x8';guard,caps=validated_build(dest)
    if(caps['target'],caps['stages'])!=('array',stages):raise ValueError('long-K check requires matching array build')
    if format not in('I32','I64'):raise ValueError('wide integer long-K format required')
    if not 1<=k<=caps['k_max']:raise ValueError('K outside the hardware range')
    period=caps['feedback_issue_interval'];hop=caps['hop_latency']
    width=64 if format=='I64'else 32;fmt=4 if width==64 else 3
    minimum=1<<(width-1);rows,cols=caps['rows'],caps['cols']
    payload=' '.join(['1']+['0']*(cols-1)+[f'{minimum:x}']+['0']*(rows-1)+[f'{minimum:x}']+['0']*(cols-1))
    trace=dest/f'long_k_{format}_{k}_commands.txt'
    points=sorted(set(checkpoints or [])|{k})
    if any(not 1<=point<=k for point in points):raise ValueError('checkpoint outside K range')
    with trace.open('w')as f:
        for i in range(k):
            f.write(f'I {fmt} 0 {int(i==0)} {i} 1 {payload}\nW {period-1}\n')
            if i+1 in points:f.write(f'D\nR 0 0 0 {i+1}\n')
    limit=timeout if timeout is not None else max(900,k//2)
    try:
        result=subprocess.run([str(dest/'sim'),str(trace)],check=True,text=True,capture_output=True,timeout=limit)
    except subprocess.TimeoutExpired:
        (dest/'verification_long_k_failed.json').write_text(json.dumps(dict(passed=False,status='host_timeout',
            numerical_failure_observed=False,stages=stages,k=k,timeout_seconds=limit,
            rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256']),indent=2)+'\n')
        raise
    observed=[]
    for line in result.stdout.splitlines():
        if not line.startswith('R '):continue
        _,tag,value,flags,cycles,macs,commits=line.split();point=int(tag)
        assert point in points and point not in observed
        assert int(value,16)==point*(1<<(2*width-2))
        assert int(flags)==0 and int(macs)==point
        observed.append(point)
    assert observed==points
    # The matrix has one active row. B tokens still drain through every
    # column/row; the last row-0 PE also has arithmetic/counter latency.
    prediction=mesh_trace_cycles(caps,trace.read_text().splitlines(True))
    checked=verify_mesh_cycles(result.stdout,prediction)
    predicted=prediction['final_cycles']
    report={'stages':stages,'k':k,'format':format,'cycles':int(cycles),
      'verified_checkpoints':observed,
      'cycle_model_prediction':predicted,'observation_cycles':caps.get('array_read_latency',0),'passed':True,'rtl_sha256':caps['rtl_sha256'],
      'simulator_sha256':caps['simulator_sha256']}
    if format=='I64':(dest/'verification_long_k.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    (dest/f'verification_long_k_{format}_{k}.json').write_text(json.dumps(report,indent=2)+'\n')
    (dest/f'long_k_{format}_{k}.log').write_text(result.stdout+result.stderr)
    (dest/'verification_long_k_failed.json').unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stages',type=int,default=2)
    p.add_argument('--k',type=int,default=65536)
    p.add_argument('--timeout',type=int,help='host runtime limit; large K requires more time on the full array')
    p.add_argument('--build-dir')
    p.add_argument('--format',choices=['I32','I64'],default='I64')
    p.add_argument('--checkpoints',nargs='+',type=int)
    a=p.parse_args();check(a.stages,a.k,a.timeout,a.build_dir,a.format,a.checkpoints)
