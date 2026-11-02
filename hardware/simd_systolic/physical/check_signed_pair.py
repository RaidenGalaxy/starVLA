#!/usr/bin/env python3
"""Prove small signed carry-save products against independent multiplication."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from prepare_analog import sha


def exhaustive(rtl,out,width,result_width,primitive,terms):
    directory=out/f'w{width}_r{result_width}_exhaustive';directory.mkdir()
    probe=directory/'probe.sv';bench=directory/'bench.cpp'
    names=['left','right','extra2','extra3'][:terms]
    ports=','.join(f'.term{i}({name})'for i,name in enumerate(names))
    probe.write_text(f'''module PairProbe(input [{width-1}:0] a,b,
output [{result_width-1}:0] {','.join(names)});
{primitive} #(.W({width}),.R({result_width})) dut(.a(a),.b(b),{ports});
endmodule
''')
    bench.write_text('''#include "VPairProbe.h"
#include <cstdint>
#include <iostream>
int main(){VPairProbe d;unsigned count=0;
for(unsigned a=0;a<LIMIT;a++)for(unsigned b=0;b<LIMIT;b++){
d.a=a;d.b=b;d.eval();
int sa=a<LIMIT/2?int(a):int(a)-LIMIT,sb=b<LIMIT/2?int(b):int(b)-LIMIT;
uint32_t expected=uint32_t(sa*sb)&MASK,actual=(ACTUAL_SUM)&MASK;
if(actual!=expected){std::cerr<<a<<" "<<b<<" "<<actual<<" "<<expected<<std::endl;return 1;}++count;}
std::cout<<count<<std::endl;return 0;}
'''.replace('ACTUAL_SUM','+'.join(f'uint32_t(d.{n})'for n in names)).replace('LIMIT',str(1<<width)).replace('MASK',str((1<<result_width)-1)+'U'))
    with(directory/'build.log').open('w')as stream:
        subprocess.run(['verilator','--cc','--exe','--build','-j','4','-Wno-fatal','--top-module','PairProbe',
            '--Mdir',str(directory/'obj'),str(rtl.resolve()),str(probe),str(bench),'-o',str(directory/'sim')],
            check=True,stdout=stream,stderr=subprocess.STDOUT)
    result=subprocess.run([str(directory/'sim')],capture_output=True,text=True,check=True)
    (directory/'simulation.log').write_text(result.stdout+result.stderr)
    if int(result.stdout.strip())!=(1<<(2*width)):raise RuntimeError('incomplete exhaustive signed-product check')
    return dict(width=width,result_width=result_width,passed=True,method='exhaustive actual RTL simulation',
        input_pairs=1<<(2*width),simulator_sha256=sha(directory/'sim'),bench_sha256=sha(bench),
        log_sha256=sha(directory/'simulation.log'))


def run(args):
    out=args.out.resolve()
    if out.exists():raise ValueError('preserve earlier product proofs')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'proof_driver.py');checks=[]
    for width in(2,3,4,7,8):
        for result_width in(2*width,2*width+5):
            if width==8:
                checks.append(exhaustive(args.rtl,out,width,result_width,args.primitive,args.terms));continue
            name=f'w{width}_r{result_width}';miter=out/(name+'.sv');script=out/(name+'.ys')
            names=['left','right','extra2','extra3'][:args.terms]
            ports=','.join(f'.term{i}({name})'for i,name in enumerate(names))
            miter.write_text(f'''module PairMiter(input [{width-1}:0] a,b,output equivalent);
wire [{result_width-1}:0] {','.join(names)};
{args.primitive} #(.W({width}),.R({result_width})) dut(.a(a),.b(b),{ports});
wire signed [{result_width-1}:0] reference=$signed(a)*$signed(b);
assign equivalent=(({'+'.join(names)})==reference);
endmodule
''')
            script.write_text(f'read_verilog -sv {args.rtl.resolve()} {miter}\nhierarchy -check -top PairMiter\nproc\nflatten\nopt_clean\nsat -prove equivalent 1 -verify\n')
            log=out/(name+'.log')
            with log.open('w')as stream:
                result=subprocess.run(['yosys','-Q','-T','-s',str(script)],stdout=stream,stderr=subprocess.STDOUT,timeout=120)
            checks.append(dict(width=width,result_width=result_width,passed=result.returncode==0,method='SAT',log_sha256=sha(log)))
            if result.returncode:raise RuntimeError('signed product proof failed: '+name)
    report=dict(passed=True,checks=checks,rtl_sha256=sha(args.rtl),primitive=args.primitive,carry_save_terms=args.terms,
        scope='all signed input pairs: SAT at W=2,3,4,7 and exhaustive RTL at W=8; two result widths each; not a64-bit formal proof')
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rtl',type=Path,required=True)
    p.add_argument('--primitive',choices=['SimdF2SignedBoothPair','SimdF2SignedBaughPair'],default='SimdF2SignedBoothPair')
    p.add_argument('--terms',type=int,choices=[2,4],default=2)
    p.add_argument('--out',type=Path,required=True);run(p.parse_args())
