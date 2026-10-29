#!/usr/bin/env python3
"""Check every dependent INT64 result through Kmax, including high carry bits."""
import argparse
import json
from pathlib import Path
import subprocess
from runtime import validated_build

ROOT=Path(__file__).resolve().parents[1]


def check(args):
    build=args.build_dir.resolve()if args.build_dir else ROOT/f'.build/pe_{args.stages}_1x1'
    guard,caps=validated_build(build)
    if(caps['target'],caps['stages'])!=('pe',args.stages):raise ValueError('PE build scope differs from requested test')
    if not 1<=args.k<=caps['k_max']:raise ValueError('K outside the supported range')
    fmt=getattr(args,'format','I64');width=64 if fmt=='I64'else 32
    format_code=4 if width==64 else 3;minimum=1<<(width-1);product=1<<(2*width-2)
    trace=build/f'pe_long_k_{fmt}_{args.k}.txt'
    with trace.open('w')as stream:
        for tag in range(args.k):stream.write(f'{format_code} 1 {minimum:x} {minimum:x} 1 0 {int(tag==0)} {tag}\n')
    result=subprocess.run([str(build/'sim'),str(trace),'--no-stall'],capture_output=True,text=True,timeout=args.timeout)
    (build/f'pe_long_k_{fmt}_{args.k}.log').write_text(result.stdout+result.stderr)
    if fmt=='I64':(build/f'pe_long_k_{args.k}.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(result.stdout[-2000:]+result.stderr)
    latency=caps['stages']+caps['pe_state_read_latency']+caps['pe_commit_latency']
    previous=None;previous_issue=None;previous_commit=None;count=0;final=None;latencies=set()
    registered=caps.get('registered_context_admission',False)
    for line in result.stdout.splitlines():
        if line.startswith('O '):
            _,tag,value,flags,overflow,start,end=line.split();tag=int(tag);start=int(start);end=int(end)
            if tag!=count or int(value,16)!=(tag+1)*product or int(flags)or int(overflow):
                raise AssertionError('dependent INT64 result, flags or tag mismatch')
            if registered:
                predicted_issue=max(start+1,previous_commit+2 if previous_commit is not None else 0)
                predicted_end=predicted_issue+latency-1
                expected_start=0 if tag==0 else 1 if tag==1 else previous_issue
                if start!=expected_start or end!=predicted_end:
                    raise AssertionError('registered dependency admission or state commit differs from independent calendar')
                if previous_issue is not None and predicted_issue-previous_issue!=latency+1:
                    raise AssertionError('registered context feedback interval changed')
                previous_issue=predicted_issue;previous_commit=predicted_end
            else:
                if end-start!=latency:raise AssertionError('unexpected dependent response latency')
                if previous is not None and start-previous!=latency+1:raise AssertionError('scoreboard feedback interval changed')
            latencies.add(end-start)
            previous=start;count+=1;final=end
        elif line.startswith('P '):
            _,ctx,value,flags=line.split()
            if int(value,16)!=(args.k*product if int(ctx)==0 else 0)or int(flags):
                raise AssertionError('drained context state differs from returned results')
    if count!=args.k:raise AssertionError('missing dependent results')
    report=dict(passed=True,scope='pe',format=fmt,stages=args.stages,k=args.k,
        verified_intermediate_results=count,final_accumulator_bits=(args.k*product).bit_length(),
        compute_cycles=final+1,response_latency_cycles=latency,
        registered_context_admission=registered,observed_response_latency_cycles=sorted(latencies),
        observed_feedback_interval_cycles=latency+1,scheduler_feedback_interval_cycles=caps['feedback_issue_interval'],
        comparison='every response and drained context checked against exact growing integer sum',
        rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'])
    if fmt=='I64':(build/'verification_long_k_pe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    (build/f'verification_long_k_pe_{fmt}_{args.k}.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stages',type=int,choices=[2],default=2)
    parser.add_argument('--build-dir',type=Path);parser.add_argument('--k',type=int,default=65536)
    parser.add_argument('--format',choices=['I32','I64'],default='I64')
    parser.add_argument('--timeout',type=int,default=3600);check(parser.parse_args())
