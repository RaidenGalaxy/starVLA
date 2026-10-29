#!/usr/bin/env python3
"""Exercise nonzero SIMD overflow and FP exception counts through the mesh."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from runtime import validated_build
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from simd_eval.reference import Format,Op,WIDTHS,ACC_WIDTHS,ENCODINGS,operate,pack,to_bits
from simd_eval.cycle_model import mesh_trace_cycles,verify_mesh_cycles


def make_vectors(caps,pipelined_read=False):
    rows,cols=caps['rows'],caps['cols'];commands=[];expected={};events={};tag=0
    totals=[0,0,0,0] # active MAC lanes, PE completions, overflow lanes, exceptional PE operations
    cases=[]
    for fmt in list(Format)[:5]:
        aw=ACC_WIDTHS[fmt]
        for negative in (False,True):
            initial=(1<<(aw-1))if negative else(1<<(aw-1))-1
            cases.append((fmt,initial,((1<<WIDTHS[fmt])-1)if negative else 1,1))
    cases.extend([(Format.F4,0x80000000,8,2),(Format.F4,0,1,1),(Format.F4,0x7f7fffff,7,7)])
    for fmt in list(Format)[6:]:
        e,m=ENCODINGS[fmt];inf=((1<<e)-1)<<m
        half,two,one=to_bits(.5,fmt),to_bits(2,fmt),to_bits(1,fmt)
        cases.extend([(fmt,0,0,inf),(fmt,0,inf-1,two),(fmt,0,1,half),(fmt,0,inf|1,one)])
    for fmt,initial,a,b in cases:
        w=WIDTHS[fmt];aw=ACC_WIDTHS[fmt];lanes=64//w
        seed=pack([initial]*lanes,aw)
        masks=[((1<<lanes)-1)if c%2==0 else sum(1<<i for i in range(0,lanes,2))for c in range(cols)]
        # Every context starts near an INT boundary or at an FP zero. Its
        # second update reuses the state, distinguishing events from stickies.
        state={(ctx,r,c):(seed,0)for ctx in range(16)for r in range(rows)for c in range(cols)}
        commands.append('D\n')
        for ctx,r,c in state:commands.append(f'S {ctx} {r} {c} 0 {seed:0128x}\n')
        for iteration in range(2):
            for ctx in range(16):
                bb=[pack([b]*lanes,w)]*cols
                commands.append(f'I {int(fmt)} {ctx} 0 {tag} {(1<<rows)-1:x} '+
                    ' '.join(f'{v:x}'for v in masks+[a]*rows+bb)+'\n');tag+=1
                for r in range(rows):
                    for c in range(cols):
                        old,sticky=state[ctx,r,c]
                        value,flags,overflow=operate(pack([a]*lanes,w),bb[c],old,fmt,Op.MAC,masks[c])
                        state[ctx,r,c]=(value,sticky|flags)
                        totals[0]+=masks[c].bit_count();totals[1]+=1
                        totals[2]+=overflow.bit_count();totals[3]+=int(bool(flags))
            commands.append('D\n')
        for(ctx,r,c),value in state.items():
            commands.append(f'{"P"if pipelined_read else"R"} {ctx} {r} {c} {tag}\n');expected[tag]=value;tag+=1
        commands.append(f'E {tag}\n');events[tag]=tuple(totals);tag+=1
    assert totals[2]>0 and totals[3]>0
    return commands,expected,events


def verify(stdout,expected,events):
    seen=set();snapshots=set()
    for line in stdout.splitlines():
        fields=line.split()
        if fields[:1]==['R']:
            tag=int(fields[1]);actual=(int(fields[2],16),int(fields[3]))
            if tag in seen or actual!=expected[tag]:raise AssertionError(('state',tag,actual,expected.get(tag)))
            seen.add(tag)
        elif fields[:1]==['E']:
            tag=int(fields[1]);actual=tuple(map(int,fields[3:]))
            if tag in snapshots or actual!=events[tag]:raise AssertionError(('events',tag,actual,events.get(tag)))
            snapshots.add(tag)
    if seen!=expected.keys()or snapshots!=events.keys():raise AssertionError('missing array state or counter observations')
    return dict(reads=len(seen),counter_snapshots=len(snapshots),macs=max(v[0]for v in events.values()),
        completions=max(v[1]for v in events.values()),overflow_events=max(v[2]for v in events.values()),
        exception_operations=max(v[3]for v in events.values()))


def check(directory,pipelined_read=False,timeout=1800):
    guard,caps=validated_build(directory)
    if caps['target']!='array':raise ValueError('array simulator required')
    if timeout<=0:raise ValueError('timeout must be positive')
    commands,expected,events=make_vectors(caps,pipelined_read);trace=directory/'event_vectors.txt';trace.write_text(''.join(commands))
    result=subprocess.run([str(directory/'sim'),str(trace)],capture_output=True,text=True,timeout=timeout)
    log=directory/'event_simulation.log';log.write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(result.stdout[-2000:]+result.stderr)
    cycle_counts=verify_mesh_cycles(result.stdout,mesh_trace_cycles(caps,commands))
    # Exclude observation times when comparing arithmetic across pipeline
    # depths; retain every result, flag and event-counter observation.
    semantic=[]
    for line in result.stdout.splitlines():
        fields=line.split()
        if fields[:1]==['R']:semantic.append(' '.join(fields[:4]+fields[5:]))
        elif fields[:1]==['E']:semantic.append(' '.join(fields[:2]+fields[3:]))
    report=dict(passed=True,**verify(result.stdout,expected,events),stages=caps['stages'],rows=caps['rows'],cols=caps['cols'],
        int_signed_boundaries=True,tail_lane_masks=True,all_contexts=16,fp_invalid_overflow_underflow=True,pipelined_read=pipelined_read,
        all_formats=True,test_suite_version=2,independent_cycle_model=True,
        cycle_observations_checked=cycle_counts['reads']+cycle_counts['markers'],
        final_bridge_cycle_prediction=cycle_counts['final_bridge_cycle_prediction'],
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        semantic_output_sha256=hashlib.sha256(('\n'.join(semantic)+'\n').encode()).hexdigest(),
        simulation_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256((ROOT/'python/simd_eval/reference.py').read_bytes()).hexdigest(),
        test_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'])
    (directory/'verification_events.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--pipelined-read',action='store_true');parser.add_argument('--timeout',type=int,default=1800)
    args=parser.parse_args();check(args.build_dir.resolve(),args.pipelined_read,args.timeout)
