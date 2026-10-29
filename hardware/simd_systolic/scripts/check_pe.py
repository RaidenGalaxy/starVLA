#!/usr/bin/env python3
"""Independent per-context oracle with long backpressure and dependent issues."""
import argparse,json,random,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from simd_eval.reference import Format,Op,WIDTHS,ENCODINGS,operate,pack,to_bits
from runtime import validated_build


def make_vectors():
    rng=random.Random(822);commands=[];expected={};acc=[0]*16;sticky=[0]*16;tag=0
    for fmt in Format:
        for round in range(30):
            for ctx in rng.sample(range(16),16):
                clear=round==0 or rng.randrange(10)==0
                if clear:acc[ctx]=0;sticky[ctx]=0
                op=Op(rng.randrange(8 if fmt<Format.F4 else 4));a=rng.getrandbits(64);b=rng.getrandbits(64);mask=rng.getrandbits(16)
                value,flags,overflow=operate(a,b,acc[ctx],fmt,op,mask)
                acc[ctx]=value;sticky[ctx]|=flags;expected[tag]=(value,flags,overflow)
                commands.append(f'{int(fmt)} {int(op)} {a:x} {b:x} {mask:x} {ctx} {int(clear)} {tag}\n');tag+=1
        if fmt>=Format.F4:
            w=WIDTHS[fmt];lanes=64//w;sign=1<<(w-1)
            values=[0,sign,to_bits(1,fmt),to_bits(-1,fmt),1]
            if fmt==Format.F4:values.append(7)
            else:
                e,f=ENCODINGS[fmt];infinity=((1<<e)-1)<<f
                values += [infinity-1,infinity,infinity|sign,infinity|(1<<(f-1)),infinity|1]
            for operation in (Op.MUL,Op.MAC,Op.ADD,Op.SUB):
                for x in values:
                    for y in values:
                        a=pack([x]*lanes,w);b=pack([y]*lanes,w);mask=(1<<lanes)-1
                        value,flags,overflow=operate(a,b,0,fmt,operation,mask)
                        expected[tag]=(value,flags,overflow);acc[0]=value;sticky[0]=flags
                        commands.append(f'{int(fmt)} {int(operation)} {a:x} {b:x} {mask:x} 0 1 {tag}\n');tag+=1
    return commands,expected,acc,sticky


def verify_output(stdout,expected,acc,sticky,minimum):
    seen=set();states={}
    for line in stdout.splitlines():
        if line.startswith('O '):
            _,tag,value,flags,overflow,start,end=line.split();tag=int(tag)
            if tag in seen:raise AssertionError('duplicate PE response tag')
            actual=(int(value,16),int(flags),int(overflow))
            if actual!=expected[tag]:raise AssertionError((tag,actual,expected[tag]))
            if int(end)-int(start)<minimum:raise AssertionError('PE result committed before its pipeline latency')
            seen.add(tag)
        if line.startswith('P '):
            _,ctx,value,flags=line.split();states[int(ctx)]=(int(value,16),int(flags))
    assert len(seen)==len(expected)
    assert states=={ctx:(acc[ctx],sticky[ctx])for ctx in range(16)}
    return len(seen)


def dependent_response_overlap(stdout,commands):
    """Count dependent issues accepted while their prior response is held."""
    timing={int(fields[1]):(int(fields[5]),int(fields[6]))for line in stdout.splitlines()
            if (fields:=line.split())and fields[0]=='O'}
    previous={};overlaps=0
    for command in commands:
        fields=command.split();ctx,tag=int(fields[5]),int(fields[7]);start,end=timing[tag]
        if ctx in previous and start<previous[ctx]:overlaps+=1
        previous[ctx]=end
    return overlaps


def check(stages,build_dir=None):
    dest=Path(build_dir)if build_dir else ROOT/f'.build/pe_{stages}_1x1';guard,caps=validated_build(dest)
    if caps['stages']!=stages:raise ValueError('build pipeline depth differs from requested check')
    commands,expected,acc,sticky=make_vectors()
    trace=dest/'pe_vectors.txt';trace.write_text(''.join(commands))
    proc=subprocess.run([str(dest/'sim'),str(trace)],capture_output=True,text=True,timeout=300)
    if proc.returncode:raise RuntimeError(proc.stdout[-2000:]+proc.stderr)
    (dest/'pe_simulation.log').write_text(proc.stdout+proc.stderr)
    minimum=stages+caps.get('pe_commit_latency',int(stages!=2))+caps.get('pe_state_read_latency',0)
    count=verify_output(proc.stdout,expected,acc,sticky,minimum)
    overlaps=dependent_response_overlap(proc.stdout,commands)
    if caps.get('fixed_state_commit')and not overlaps:raise AssertionError('fixed commit test did not exercise a dependent issue before response consumption')
    report=dict(passed=True,stages=stages,vectors=count,contexts=16,dependent_issues=True,
                long_result_stalls=True,sticky_flags=True,rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'])
    report['dependent_issues_before_prior_response_consumption']=overlaps
    (dest/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stages',type=int,default=2);p.add_argument('--build-dir');a=p.parse_args();check(a.stages,a.build_dir)
