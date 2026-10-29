#!/usr/bin/env python3
"""Require identical actual observations and clocks across host schedulers."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
from runtime import validated_build
from check_array import make_vectors,verify_output
from check_array_events import make_vectors as event_vectors,verify as verify_events


def check(builds,out,timeout):
    guards=[];caps=[]
    for build in builds:
        guard,cap=validated_build(build);guards.append(guard);caps.append(cap)
    for cap in caps:
        if cap['target']!='array':raise ValueError('array simulators required')
        for key in ('rtl_sha256','stages','rows','cols'):
            if cap[key]!=caps[0][key]:raise ValueError('host threading comparison requires identical RTL and dimensions')
    workers=[c.get('simulation_threads',1)for c in caps]
    if any(c.get('simulation_root_threads',1)!=n for c,n in zip(caps,workers)):
        raise ValueError('root scheduling must actually enable the requested workers; rebuild with the corrected --threads setting')
    if 1 not in workers or max(workers)==1:raise ValueError('single and multiple host workers required')
    out.mkdir(parents=True,exist_ok=True);results=[]
    commands,expected,macs=make_vectors(caps[0],k=17,pipelined_read=True)
    events,states,counters=event_vectors(caps[0],pipelined_read=True)
    for name,trace_lines,verify in [('matrix',commands,lambda text:verify_output(text,expected,macs)),
                                    ('events',events,lambda text:verify_events(text,states,counters))]:
        trace=out/(name+'.txt');trace.write_text(''.join(trace_lines));baseline=None
        for build,cap,threads in zip(builds,caps,workers):
            begin=time.monotonic();proc=subprocess.run([str(build/'sim'),str(trace)],check=True,capture_output=True,text=True,timeout=timeout)
            elapsed=time.monotonic()-begin;counts=verify(proc.stdout)
            log=out/f'{name}_mt{threads}.log';log.write_text(proc.stdout+proc.stderr)
            if baseline is None:baseline=proc.stdout
            elif proc.stdout!=baseline:raise AssertionError('host workers changed result, flag, counter, or observation cycle')
            results.append(dict(case=name,simulation_threads=threads,host_wall_seconds=elapsed,
                observations=counts,rtl_sha256=cap['rtl_sha256'],simulator_sha256=cap['simulator_sha256'],
                output_sha256=hashlib.sha256(proc.stdout.encode()).hexdigest(),trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest()))
    report=dict(passed=True,scope='host_simulation_thread_equivalence',all_formats=True,
        exact_stdout_including_cycles=True,host_times_are_not_hardware_speedup=True,timing_certification=False,
        results=results)
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,workers=workers,stages=caps[0]['stages'],rows=caps[0]['rows'],cols=caps[0]['cols'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--builds',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--timeout',type=int,default=7200);args=parser.parse_args()
    check([p.resolve()for p in args.builds],args.out.resolve(),args.timeout)
