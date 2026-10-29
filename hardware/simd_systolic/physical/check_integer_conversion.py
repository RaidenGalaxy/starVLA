#!/usr/bin/env python3
"""Sweep integer-to-FP32 leading positions and RNE ties against the oracle."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
from runtime import validated_build
from simd_eval.reference import Format,Op,ACC_WIDTHS,WIDTHS,operate,pack
from prepare_analog import sha


def run(build,out):
    build=build.resolve();guard,caps=validated_build(build)
    if caps['target']!='alu'or caps['stages']!=2:raise ValueError('requires the actual two-stage ALU model')
    out=out.resolve()
    if out.exists():raise ValueError('preserve prior conversion checks')
    out.mkdir(parents=True)
    vectors=[];expected=[];positions={}
    for fmt in list(Format)[:5]:
        width=ACC_WIDTHS[fmt];lanes=64//WIDTHS[fmt];limit=1<<(width-1)
        numbers={0,1,-1,limit-1,-limit};covered=[]
        for position in range(width):
            base=1<<position;values={base-1,base,base+1}
            if position>=24:
                ulp=1<<(position-23)
                for parity in(0,1):
                    for offset in(-1,0,1):values.add(base+parity*ulp+ulp//2+offset)
            for magnitude in values:
                for sign in(-1,1):
                    value=sign*magnitude
                    if -limit<=value<limit:numbers.add(value)
            covered.append(position)
        positions[fmt.name]=covered
        for number in sorted(numbers):
            c=pack([number&((1<<width)-1)]*lanes,width)
            for mask in sorted({0,(1<<lanes)-1,0xaaaa&((1<<lanes)-1)}):
                tag=len(vectors);vectors.append(f'{int(fmt)} 7 0 0 {c:x} {mask:x} {tag%16} {tag}\n')
                expected.append(operate(0,0,c,fmt,Op(7),mask))
    trace=out/'vectors.txt';trace.write_text(''.join(vectors));records=[]
    for mode in('continuous','stalled_idle'):
        env={k:v for k,v in os.environ.items()if k!='SIMD_ALU_IDLE_GAPS'}
        if mode=='stalled_idle':env['SIMD_ALU_IDLE_GAPS']='1'
        result=subprocess.run([str(build/'sim'),str(trace)]+(['101']if mode=='stalled_idle'else[]),
            env=env,capture_output=True,text=True,check=True,timeout=120)
        log=out/(mode+'.log');log.write_text(result.stdout+result.stderr);seen=set()
        for line in result.stdout.splitlines():
            if not line or not line[0].isdigit():continue
            tag,data,flags,overflow,illegal,start,end=line.split();tag=int(tag)
            if tag in seen or tag>=len(expected):raise AssertionError('invalid conversion response tag')
            if(int(data,16),int(flags,16),int(overflow,16))!=expected[tag]or int(illegal):
                raise AssertionError('conversion mismatch: '+vectors[tag].strip())
            if mode=='continuous'and int(end)-int(start)!=2:raise AssertionError('conversion latency is not F2')
            seen.add(tag)
        if len(seen)!=len(expected):raise AssertionError('missing conversion outputs')
        records.append(dict(mode=mode,vectors=len(vectors),log_sha256=sha(log)))
    report=dict(passed=True,kind='independent_integer_conversion_rounding_sweep',pipeline_stages=2,
        integer_formats=5,leading_positions=positions,round_to_nearest_even_ties=True,
        masks_and_both_signs=True,checks=records,rtl_sha256=caps['rtl_sha256'],
        simulator_sha256=caps['simulator_sha256'],trace_sha256=sha(trace),
        reference_sha256=sha(ROOT/'python/simd_eval/reference.py'),driver_sha256=sha(__file__),
        physical_frequency_certified=False)
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.build_dir,a.out)
