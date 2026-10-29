#!/usr/bin/env python3
"""Inject actual memory-response errors and check descriptor recovery."""
import argparse
import json
import subprocess
from pathlib import Path
from runtime import validated_build

ROOT=Path(__file__).resolve().parents[1]
import sys as _architecture_sys
_architecture_sys.path.insert(0,str(ROOT/'python'))
from simd_eval.architecture import CORE_PS, MEMORY_PS


def check(stages=2,dram=False,build_dir=None,out=None):
    dest=Path(build_dir)if build_dir else ROOT/f'.build/gemm_{stages}_2x2'
    guard,caps=validated_build(dest)
    if(caps['target'],caps['stages'])!=('gemm',stages):raise ValueError('fault check requires the matching GEMM build')
    out=Path(out)if out else ROOT/f'.build/gemm_faults_{stages}_{"dram"if dram else"ideal"}'
    out.mkdir(parents=True,exist_ok=True)
    for direction,address in [('r',0x10000000),('w',0x80000000)]:
        commands=f'''L 268435456 00000000000000000000000000000003
L 805306368 00000000000000000000000000000002
F {direction} {address}
G 0 1 1 1 268435456 805306368 2147483651 8 8 4 17
G 0 1 1 1 268435456 805306368 2147483907 8 8 4 18
'''
        trace=out/f'{direction}.txt';trace.write_text(commands)
        process=subprocess.run([str(dest/'sim'),str(trace),str(CORE_PS),str(MEMORY_PS),
            str(ROOT/'configs/ddr4.ini')if dram else'-',str(out/direction),'207'],
            capture_output=True,text=True,timeout=120)
        (out/f'{direction}.log').write_text(process.stdout+process.stderr)
        if process.returncode:raise RuntimeError(process.stdout[-2500:]+process.stderr)
        errors=[];done=[];written={}
        for line in process.stdout.splitlines():
            if line.startswith('F '):errors.append(tuple(map(int,line.split()[1:])))
            elif line.startswith('D '):
                fields=list(map(int,line.split()[1:]));done.append(fields[0])
                if fields[1]!=0 or fields[3]!=1 or fields[4]!=0:raise AssertionError('arithmetic/counter mismatch')
            elif line.startswith('W '):
                _,address,value=line.split();written[int(address)]=int(value)
        expected={base+offset:(6 if offset==0 else 0)for base in(2147483651,2147483907)for offset in range(4)}
        if errors!=[(17,1)]or done!=[17,18]or written!=expected:
            raise AssertionError(f'error was lost or subsequent descriptor did not recover: {errors}, {done}, {written}')
    report=dict(passed=True,stages=stages,rows=caps['rows'],cols=caps['cols'],dram=dram,descriptors=4,
        read_response_error=True,write_response_error=True,subsequent_descriptor_recovery=True,
        rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'])
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stages',type=int,default=2);p.add_argument('--dram',action='store_true')
    p.add_argument('--build-dir');p.add_argument('--out')
    check(**vars(p.parse_args()))
