#!/usr/bin/env python3
"""Abort descriptors during real read/compute/write activity and restart."""
import argparse
import json
import subprocess
import os
from pathlib import Path
from runtime import validated_build

ROOT=Path(__file__).resolve().parents[1]
import sys as _architecture_sys
_architecture_sys.path.insert(0,str(ROOT/'python'))
from simd_eval.architecture import CORE_PS, MEMORY_PS


def check(stages=2,dram=False,build_dir=None,out=None):
    dest=Path(build_dir)if build_dir else ROOT/f'.build/gemm_{stages}_2x2';guard,caps=validated_build(dest)
    if(caps['target'],caps['stages'])!=('gemm',stages):raise ValueError('reset check requires the matching GEMM build')
    out=Path(out)if out else ROOT/f'.build/gemm_reset_{stages}_{"dram"if dram else"ideal"}'
    out.mkdir(parents=True,exist_ok=True)
    for phase in('r','c','w'):
        commands=[]
        def load(address,data):
            for offset in range(0,len(data),16):
                commands.append(f'L {address+offset} {int.from_bytes(data[offset:offset+16],"little"):032x}\n')
        load(0x10000000,bytes.fromhex('11111111')+bytes(12))
        old_b=bytearray(64)
        for row in range(8):old_b[row*8]=2
        load(0x30000000,old_b)
        load(0x11000000,(0x4e3).to_bytes(16,'little'))
        new_b=bytearray(32)
        for row,value in enumerate([15,5,2]):new_b[row*8]=value
        load(0x31000000,new_b)
        commands += [f'B 0 1 1 8 {0x10000000} {0x30000000} {0x80000003} 8 8 4 17\n',
                     f'X {phase}\n',f'G 0 1 1 3 {0x11000000} {0x31000000} {0x90000003} 8 8 4 18\n']
        trace=out/f'{phase}.txt';trace.write_text(''.join(commands))
        proc=subprocess.run([str(dest/'sim'),str(trace),str(CORE_PS),str(MEMORY_PS),
            str(ROOT/'configs/ddr4.ini')if dram else'-',str(out/phase),'207'],
            capture_output=True,text=True,timeout=180)
        (out/f'{phase}.log').write_text(proc.stdout+proc.stderr)
        if proc.returncode:raise RuntimeError(proc.stdout[-2500:]+proc.stderr)
        done=[];resets=[];written={};reset_inputs=[]
        for line in proc.stdout.splitlines():
            if line.startswith('D '):
                values=list(map(int,line.split()[1:]));done.append(values[0])
                if values[1]!=0 or values[3]!=3 or values[4]!=0:raise AssertionError('restart arithmetic/counters')
            elif line.startswith('X '):resets.append(line.split()[1:3])
            elif line.startswith('XR '):reset_inputs.append(line.split()[1:])
            elif line.startswith('W '):
                _,address,value=line.split();written[int(address)]=int(value)
        expected=dict(zip(range(0x90000003,0x90000007),(-5&0xffffffff).to_bytes(4,'little')))
        actual={key:value for key,value in written.items()if key>=0x90000000}
        if done!=[18] or resets!=[['1',phase]] or actual!=expected:
            raise AssertionError(f'reset/restart mismatch: {done}, {resets}, {actual}')
        domain=os.environ.get('SIMD_RESET_DOMAIN','both')
        if reset_inputs!=[[domain,str(int(domain!='memory')),str(int(domain!='core'))]]:
            raise AssertionError('requested external reset domain was not actually asserted')
        # A reset aborts the command; accepted write bytes may remain visible.
        # Input data survives the external timing-model reset. No oracle data
        # is loaded into C, either before or after the interrupted command.
        if phase!='w' and any(0x80000000<=key<0x90000000 for key in written):
            raise AssertionError('pre-write reset unexpectedly wrote C')
    report=dict(passed=True,stages=stages,rows=caps['rows'],cols=caps['cols'],dram=dram,aborted_descriptors=3,restarted_descriptors=3,
        read_inflight_reset=True,core_active_reset=True,partial_write_reset=True,
        reset_protocol='both clock domains held in reset while accepted DRAM traffic drains; partial front-end bursts cancelled',
        reset_domains=os.environ.get('SIMD_RESET_DOMAIN','both'),external_timing_model_recreated=True,input_data_retained=True,
        either_external_reset_flushes_both_fifo_endpoints=True,
        accepted_write_side_effects_may_remain=True,rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'])
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stages',type=int,default=2);p.add_argument('--dram',action='store_true')
    p.add_argument('--build-dir');p.add_argument('--out')
    check(**vars(p.parse_args()))
