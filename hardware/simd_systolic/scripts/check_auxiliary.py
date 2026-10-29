#!/usr/bin/env python3
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
import sys as _architecture_sys
_architecture_sys.path.insert(0,str(ROOT/'python'))
from simd_eval.architecture import CORE_PS, MEMORY_PS
from runtime import validated_build
from simd_eval.reference import Format,Op,pack,to_bits,operate

def check(stages=2,dram=False,build_dir=None):
    dest=Path(build_dir)if build_dir else ROOT/f'.build/system_{stages}_2x2'
    build_guard,caps=validated_build(dest)
    if(caps['target'],caps['stages'],caps['rows'],caps['cols'])!=('system',stages,2,2):raise ValueError('build scope differs from auxiliary check')
    suffix=f'_sram{caps["sram_read_latency"]}'if caps.get('sram_read_latency',1)!=1 else''
    out=ROOT/f'.build/auxiliary_{stages}_{"dram"if dram else"ideal"}{suffix}';out.mkdir(parents=True,exist_ok=True)
    f=lambda x:to_bits(x,Format.F32)
    scale=pack([f(1.5),f(2),f(1),f(4)],32)
    integers=pack([-3,5]+[0]*14,32)
    commands=f'''L 2 0 {scale:032x}
A 9 0 0 0 0 3 1 0 0 0 1 0
A 0 7 0 0 {integers:x} ffff 2 0 0 0 0 0
A 9 1 @2:0:64 @1:0:64 0 3 3 1 0 1 0 0
A 9 1 @2:0:64 @1:0:64 0 3 4 1 0 0 0 0
A 9 1 @2:0:64 @1:0:64 0 3 5 1 8191 1 0 0
A 9 1 @2:0:64 @1:0:64 0 3 6 1 0 0 0 0
A 9 1 @2:0:64 @1:0:64 0 3 7 1 8191 0 0 0
O 8
'''
    trace=out/'commands.txt';trace.write_text(commands)
    result=subprocess.run([str(dest/'sim'),str(trace),
      str(CORE_PS),str(MEMORY_PS),str(ROOT/'configs/ddr4.ini')if dram else'-',str(out/'dram'),'207'],capture_output=True,text=True,timeout=60)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    expect={1:pack([f(3),f(4)],32),2:pack([f(-3),f(5)]+[0]*14,32),
            3:pack([f(-9),f(20)],32),4:pack([f(-18),f(40)],32),
            5:pack([f(-9),f(20)],32),6:pack([f(-27),f(60)],32),7:pack([f(-18),f(40)],32)}
    seen={}
    for line in result.stdout.splitlines():
        if line.startswith('R '):
            _,t,v,flags,ps=line.split();seen[int(t)]=int(v,16)
            if int(flags):raise AssertionError('unexpected exact-operation FP flags')
    assert seen==expect,(seen,expect)
    (out/'simulation.log').write_text(result.stdout+result.stderr)
    report={'stages':stages,'dram':dram,'sram_read_latency':caps.get('sram_read_latency',1),'operations':len(expect),'aggregate_highest_index':8191,'aggregate_bank_independence':True,'memory_period_ps':MEMORY_PS,'result_backpressure':True,'passed':True,'rtl_sha256':caps['rtl_sha256'],'simulator_sha256':caps['simulator_sha256']}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stages',type=int,default=2);p.add_argument('--dram',action='store_true');p.add_argument('--build-dir');a=p.parse_args();check(a.stages,a.dram,a.build_dir)
