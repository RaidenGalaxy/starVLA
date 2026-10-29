#!/usr/bin/env python3
import argparse,json,random,subprocess,sys
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from runtime import validated_build
from simd_eval.reference import Format,WIDTHS,ACC_WIDTHS,gemm,pack,to_bits
from simd_eval.cycle_model import mesh_trace_cycles,verify_mesh_cycles

def make_vectors(caps,k=17,pipelined_read=False):
    rows,cols=caps['rows'],caps['cols']
    randomizer=random.Random(318);commands=[];expected={};tag=0;expected_macs=0
    for fmt in Format:
        w=WIDTHS[fmt];aw=ACC_WIDTHS[fmt];lanes=64//w
        for contexts in (1,16):
            m=max(1,rows-1);n=cols*lanes-1 if cols*lanes>1 else 1
            def number():
                v=randomizer.randrange(-3,4)
                return (v&((1<<w)-1)) if fmt<Format.F4 else to_bits(Fraction(v,2),fmt)
            matrices=[]
            for ctx in range(contexts):
                a=[[number() for _ in range(k)]for _ in range(m)]
                b=[[number() for _ in range(n)]for _ in range(k)]
                matrices.append((a,b,gemm(a,b,fmt)))
            masks=[(1<<max(0,min(lanes,n-c*lanes)))-1 for c in range(cols)]
            # Separate two K blocks to exercise retained context accumulators.
            split=max(1,k//2)
            for begin,end in [(0,split),(split,k)]:
                for t in range(begin,end):
                    for ctx,(a,b,_) in enumerate(matrices):
                        aa=[a[r][t] if r<m else 0 for r in range(rows)]
                        bb=[pack(b[t][c*lanes:(c+1)*lanes],w)for c in range(cols)]
                        commands.append('I '+f'{int(fmt)} {ctx} {int(t==0)} {tag} {(1<<m)-1:x} '+
                            ' '.join(f'{x:x}' for x in masks+aa+bb)+'\n');tag+=1
                    gap=max(0,caps['feedback_issue_interval']-contexts)
                    if gap:commands.append(f'W {gap}\n')
                commands.append('D\n')
            expected_macs+=m*n*k*contexts
            for ctx,(_,_,out) in enumerate(matrices):
                for r in range(rows):
                    for c in range(cols):
                        value=pack(out[r][c*lanes:(c+1)*lanes],aw) if r<m else 0
                        expected[tag]=value;commands.append(f'{"P"if pipelined_read else"R"} {ctx} {r} {c} {tag}\n');tag+=1
    return commands,expected,expected_macs


def verify_output(stdout,expected,expected_macs):
    seen=set();cycles=0;macs=0
    for line in stdout.splitlines():
        if not line.startswith('R '):continue
        _,t,v,flags,cycle,mac,commits=line.split();t=int(t)
        if t in seen or t not in expected:raise AssertionError('duplicate or unexpected array read tag')
        seen.add(t)
        if int(v,16)!=expected[t]:raise AssertionError(f'read {t} expected {expected[t]:x}, got {v}')
        if int(flags):raise AssertionError('unexpected exception on exact finite array vectors')
        cycles=int(cycle);macs=int(mac)
    assert len(seen)==len(expected)
    assert macs==expected_macs,(macs,expected_macs)
    return dict(reads=len(seen),macs=macs,cycles=cycles)


def check(stages=2,rows=2,cols=2,k=17,pipelined_read=False,build_dir=None):
    dest=Path(build_dir)if build_dir else ROOT/f'.build/array_{stages}_{rows}x{cols}'
    build_guard,caps=validated_build(dest)
    if (caps['stages'],caps['rows'],caps['cols'])!=(stages,rows,cols):raise ValueError('build scope differs from requested check')
    commands,expected,expected_macs=make_vectors(caps,k,pipelined_read)
    trace=dest/'array_commands.txt';trace.write_text(''.join(commands))
    result=subprocess.run([str(dest/'sim'),str(trace)],text=True,capture_output=True)
    (dest/'array_simulation.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    counts=verify_output(result.stdout,expected,expected_macs)
    prediction=mesh_trace_cycles(caps,commands)
    cycle_counts=verify_mesh_cycles(result.stdout,prediction)
    report={'stages':stages,'rows':rows,'cols':cols,**counts,'independent_cycle_model':True,
      'cycle_observations_checked':cycle_counts['reads'],'final_bridge_cycle_prediction':prediction['final_cycles'],
      'pipelined_read':pipelined_read,'passed':True,'rtl_sha256':caps['rtl_sha256'],'simulator_sha256':caps['simulator_sha256']}
    (dest/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stages',type=int,default=2);p.add_argument('--rows',type=int,default=2)
    p.add_argument('--cols',type=int,default=2);p.add_argument('--k',type=int,default=17);p.add_argument('--pipelined-read',action='store_true');p.add_argument('--build-dir');a=p.parse_args()
    check(a.stages,a.rows,a.cols,a.k,a.pipelined_read,a.build_dir)
