#!/usr/bin/env python3
import argparse,json,random,subprocess,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from runtime import validated_build
from simd_eval.reference import Format,Op,WIDTHS,ACC_WIDTHS,operate,pack,to_bits

def make_vectors(count=100,exhaustive=False):
    r=random.Random(207);vs=[];expected=[]
    def add(fmt,op,a,b,c=0,mask=65535):
        tag=len(vs);vs.append(f'{int(fmt)} {int(op)} {a:x} {b:x} {c:x} {mask:x} {tag%16} {tag}\n')
        expected.append(operate(a,b,c,fmt,op,mask))
    for f in Format:
        w=WIDTHS[f];aw=ACC_WIDTHS[f];lanes=64//w
        ops=range(8 if f<Format.F4 else 4)
        edge=[0,1,(1<<w)-1,1<<(w-1),(1<<(w-1))-1]
        for op in ops:
            for a in edge:
                for b in edge:
                    add(f,op,pack([a]*lanes,w),pack([b]*lanes,w))
            for _ in range(count):
                add(f,op,r.getrandbits(64),r.getrandbits(64),r.getrandbits(aw*lanes),r.getrandbits(lanes))
        if f in (Format.I4,Format.F4) or (exhaustive and f==Format.F8):
            for a in range(1<<w):
                for b in range(1<<w): add(f,Op.MAC,a,b,mask=1)
    for f,minimum,previous,quarter in [(Format.F32,0x800000,0x3f7fffff,0x3e800000),
                                      (Format.F64,0x10000000000000,0x3fefffffffffffff,0x3fd0000000000000)]:
        add(f,Op.MUL,minimum,previous,mask=1)
        add(f,Op.MAC,(1<<(WIDTHS[f]-1))|1,quarter,minimum,mask=1)
    return vs,expected


def check(stages=2,count=100,stall=False,exhaustive=False,build_dir=None,idle_gaps=False):
    vs,expected=make_vectors(count,exhaustive)
    dest=Path(build_dir).resolve()if build_dir else ROOT/f'.build/alu_{stages}_1x1';build_guard,caps=validated_build(dest)
    if caps['target']!='alu'or caps['stages']!=stages:raise ValueError('ALU build scope differs from requested check')
    vf=dest/'vectors.txt';vf.write_text(''.join(vs))
    command=[str(dest/'sim'),str(vf)]+(['101'] if stall else [])
    env={**os.environ,'SIMD_ALU_IDLE_GAPS':'1'}if idle_gaps else {k:v for k,v in os.environ.items()if k!='SIMD_ALU_IDLE_GAPS'}
    result=subprocess.run(command,capture_output=True,text=True,check=True,env=env)
    lines=result.stdout.splitlines();seen=set()
    for line in lines:
        if not line or not line[0].isdigit(): continue
        t,data,flags,overflow,illegal,start,end=line.split();i=int(t);seen.add(i)
        actual=(int(data,16),int(flags,16),int(overflow,16))
        if actual!=expected[i] or int(illegal):
            raise AssertionError(f'vector {i}: {vs[i].strip()} expected={tuple(hex(x) for x in expected[i])} actual={tuple(hex(x) for x in actual)}')
        if not stall and int(end)-int(start)!=stages:
            raise AssertionError(f'latency {int(end)-int(start)} != {stages}')
    if len(seen)!=len(vs):raise AssertionError('missing/duplicate outputs')
    report={'profile':'F2' if stages==2 else f'HF{stages}','vectors':len(vs),'random_seed':207,'backpressure':stall,'idle_gaps':idle_gaps,'passed':True,'rtl_sha256':caps['rtl_sha256'],'simulator_sha256':caps['simulator_sha256']}
    (dest/('verification_idle.json'if idle_gaps else'verification_stall.json' if stall else 'verification.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stages',type=int,default=2)
    p.add_argument('--count',type=int,default=100);p.add_argument('--stall',action='store_true')
    p.add_argument('--idle-gaps',action='store_true')
    p.add_argument('--exhaustive',action='store_true');p.add_argument('--build-dir',type=Path);a=p.parse_args()
    check(a.stages,a.count,a.stall,a.exhaustive,a.build_dir,a.idle_gaps)
