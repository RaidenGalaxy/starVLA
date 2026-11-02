#!/usr/bin/env python3
"""Compare actual mapped ASAP7 async/reset flops with the RTL over clock/reset events."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from run import ROOT,sha
from asap7_platform import profile


def run(out,library=None):
    out=out.resolve()
    if out.exists():raise ValueError('preserve previous asynchronous mapping proofs')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'proof_driver.py')
    mapping=ROOT/'physical/Asap7AsyncFlops.v';shutil.copy2(mapping,out/mapping.name)
    library=Path(library).resolve()if library else profile('TT')['liberty']
    source=out/'source.sv';source.write_text('''module AsyncProbe(input clock,rp,rn,data,en,output [7:0] value);
reg p0,p1,n0,n1,e0,e1,first,second;
always @(posedge clock or posedge rp)if(rp)p0<=0;else p0<=data;
always @(posedge clock or posedge rp)if(rp)p1<=1;else p1<=data;
always @(posedge clock or negedge rn)if(!rn)n0<=0;else n0<=data;
always @(posedge clock or negedge rn)if(!rn)n1<=1;else n1<=data;
always @(posedge clock or posedge rp)if(rp)e0<=0;else if(en)e0<=data;
always @(posedge clock or posedge rp)if(rp)e1<=1;else if(!en)e1<=data;
always @(posedge clock or posedge rp)if(rp)begin first<=1;second<=1;end
else begin first<=0;second<=first;end
assign value={second,first,e1,e0,n1,n0,p1,p0};
endmodule
''')
    script=out/'map.ys';script.write_text(f'''read_verilog -sv {source}
hierarchy -check -top AsyncProbe
synth -top AsyncProbe -flatten -noabc
read_liberty -lib {library}
dffunmap
techmap -map {out/mapping.name}
techmap
select -assert-none t:$_DFFSR_* t:$_DFF_*0_ t:$_DFF_*1_ t:$_DFFE_*0?_ t:$_DFFE_*1?_
dfflibmap -liberty {library}
abc -liberty {library}
clean
rename AsyncProbe AsyncMapped
write_verilog -noattr -noexpr {out/'mapped.v'}
''')
    with(out/'mapping.log').open('w')as stream:
        subprocess.run(['yosys','-Q','-T','-s',str(script)],check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
    (out/'reference.sv').write_text(source.read_text().replace('module AsyncProbe','module AsyncReference',1))
    cells=out/'cells.v'
    subprocess.run(['yosys','-Q','-T','-p',f'read_liberty -ignore_miss_func {library}; write_verilog {cells}'],check=True,stdout=subprocess.DEVNULL)
    top=out/'top.sv';top.write_text('''module AsyncMappingProbe(input clock,rp,rn,data,en,output [7:0] reference_value,mapped_value);
AsyncReference reference_unit(.*,.value(reference_value));
AsyncMapped mapped_unit(.*,.value(mapped_value));
endmodule
''')
    bench=out/'bench.cpp';bench.write_text('''#include "VAsyncMappingProbe.h"
#include <iostream>
#include <cstdint>
int main(){VAsyncMappingProbe d;unsigned checks=0;uint32_t rng=71;
auto check=[&](){d.eval();++checks;if(d.reference_value!=d.mapped_value){
std::cerr<<"ASYNC_MAPPING_MISMATCH "<<checks<<" "<<unsigned(d.reference_value)<<" "<<unsigned(d.mapped_value)<<std::endl;return false;}return true;};
d.clock=0;d.rp=0;d.rn=1;d.data=0;d.en=0;d.eval();
d.rp=1;d.rn=0;if(!check())return 1;
d.clock=1;if(!check())return 1;d.clock=0;if(!check())return 1;
d.rp=0;d.rn=1;if(!check())return 1;
for(int i=0;i<6;i++){d.clock^=1;if(!check())return 1;}
for(int i=0;i<5000;i++){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;
d.clock=(rng>>4)&1;d.rp=(rng&31)==0;d.rn=((rng>>8)&31)!=0;d.en=(rng>>16)&1;d.data=(rng>>17)&1;if(!check())return 1;}
std::cout<<"ASYNC_MAPPING_CHECKS "<<checks<<std::endl;return 0;}
''')
    with(out/'build.log').open('w')as stream:
        subprocess.run(['verilator','--cc','--exe','--build','-j','4','-Wno-fatal','--top-module','AsyncMappingProbe',
            '--Mdir',str(out/'obj'),str(cells),str(out/'reference.sv'),str(out/'mapped.v'),str(top),str(bench),'-o',str(out/'sim')],check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
    result=subprocess.run([str(out/'sim')],check=True,capture_output=True,text=True,timeout=30)
    (out/'simulation.log').write_text(result.stdout+result.stderr)
    checks=int(result.stdout.split('ASYNC_MAPPING_CHECKS ')[1].split()[0])
    assert checks==5010
    record=dict(passed=True,event_checks=checks,async_reset_values=[0,1],async_reset_polarities=['positive','negative'],
        enabled_flops=True,two_stage_reset_assert_and_release=True,liberty_sha256=sha(library),mapping_sha256=sha(mapping),
        physical_frequency_certified=False,evidence={str(p):sha(p)for p in [library,*[p for p in out.iterdir()if p.is_file()]]})
    (out/'verification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items()if k!='evidence'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--liberty',type=Path);args=p.parse_args();run(args.out,args.liberty)
