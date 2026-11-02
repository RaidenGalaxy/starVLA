#!/usr/bin/env python3
"""Prove one-cut modular addition against an independent registered sum."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from prepare_analog import sha


def run(args):
    out=args.out.resolve()
    if out.exists():raise ValueError('preserve earlier chunk-adder proofs')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'proof_driver.py');checks=[]
    for width in(3,4,7,8,9,17):
        name='w'+str(width);miter=out/(name+'.sv');script=out/(name+'.ys')
        extra=''
        if args.prefix_outputs:
            extra=f'''wire [{width-2}:0] leading,lower,expected_leading,expected_lower;
for(genvar j=0;j<{width-1};j++)begin
 if(j=={width-2})assign expected_leading[j]=reference[j];
 else assign expected_leading[j]=reference[j]&&!(|reference[{width-2}:j+1]);
 assign expected_lower[j]=|reference[j:0];
end
'''
        miter.write_text(f'''module ChunkMiter(input clock,ce,input [{width-1}:0] a,b,c,
input [1:0] bias,output equivalent);
wire [{width-1}:0] actual;reg [{width-1}:0] reference;
{extra}
SimdRegisteredChunkSum #(.W({width}),.B(8)) dut(.*,.result(actual));
always @(posedge clock)if(ce)reference<=a+b+c+bias;
assign equivalent=(actual==reference){'&&(leading==expected_leading)&&(lower==expected_lower)'if args.prefix_outputs else''};
endmodule
''')
        script.write_text(f'read_verilog -sv {args.rtl.resolve()} {miter}\nhierarchy -check -top ChunkMiter\nproc\nflatten\nopt_clean\ndffunmap\nopt_clean\nsat -seq 3 -set-def-inputs -set-at 1 ce 1 -prove-skip 1 -prove equivalent 1 -verify\n')
        log=out/(name+'.log')
        with log.open('w')as stream:
            result=subprocess.run(['yosys','-Q','-T','-s',str(script)],stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        checks.append(dict(width=width,passed=result.returncode==0,log_sha256=sha(log)))
        if result.returncode:raise RuntimeError('registered chunk-adder proof failed: '+name)
    report=dict(passed=True,checks=checks,rtl_sha256=sha(args.rtl),
        prefix_outputs_checked=args.prefix_outputs,
        scope='all operands/biases at six small widths; three clocks with initial CE=1 and arbitrary later stalls; not full FMA formal equivalence')
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rtl',type=Path,required=True)
    p.add_argument('--prefix-outputs',action='store_true')
    p.add_argument('--out',type=Path,required=True);run(p.parse_args())
