#!/usr/bin/env python3
"""Check live state reads across seed, first commit, clear, and result stalls."""
import argparse,json,subprocess
from pathlib import Path
from check_pe import make_vectors,verify_output,dependent_response_overlap,ROOT
from runtime import validated_build

def state_vectors(commands,expected,caps):
    if not caps.get('fixed_state_commit'):return ''.join(commands)
    return ''.join(command.rstrip()+f' {expected[int(command.split()[7])][0]:0128x} {expected[int(command.split()[7])][1]}\n'
                   for command in commands)

def check(directory):
    guard,caps=validated_build(directory)
    if caps['target']!='pe'or not caps.get('live_state_observation_check'):
        raise ValueError('a PE simulator with live state observation support is required')
    commands,expected,acc,sticky=make_vectors();trace=directory/'state_vectors.txt';trace.write_text(state_vectors(commands,expected,caps))
    proc=subprocess.run([str(directory/'sim'),str(trace),'--state-stress'],capture_output=True,text=True,timeout=300)
    (directory/'state_simulation.log').write_text(proc.stdout+proc.stderr)
    if proc.returncode:
        (directory/'verification_state_failed.json').write_text(json.dumps(dict(passed=False,
            returncode=proc.returncode,rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'],
            diagnostic=proc.stderr[-4000:]),indent=2)+'\n')
        raise RuntimeError(f'live PE state check failed; see {directory/"state_simulation.log"}')
    count=verify_output(proc.stdout,expected,acc,sticky,caps['stages']+caps['pe_commit_latency']+caps['pe_state_read_latency'])
    records=[list(map(int,line.split()[1:]))for line in proc.stdout.splitlines()if line.startswith('Q ')]
    if len(records)!=1 or records[0][1:3]!=[16,count]or records[0][0]<=count or records[0][3]<=0:
        raise AssertionError('missing live read/seed/commit coverage')
    observations,seeds,commits,disabled=records[0]
    overlaps=dependent_response_overlap(proc.stdout,commands)
    if caps.get('fixed_state_commit')and not overlaps:raise AssertionError('live state test did not exercise response-independent feedback')
    admission=[list(map(int,line.split()[1:]))for line in proc.stdout.splitlines()if line.startswith('A ')]
    registered=caps.get('registered_context_admission',False)
    if registered and(len(admission)!=1 or len(admission[0])!=2 or min(admission[0])<=0):
        raise AssertionError('registered admission test did not exercise queued dependencies')
    report=dict(passed=True,stages=caps['stages'],vectors=count,live_state_observations=observations,
        seeded_contexts=seeds,commits=commits,disabled_read_clocks=disabled,
        dependent_issues_before_prior_response_consumption=overlaps,
        read_latency=caps['pe_read_latency'],fixed_state_commit=caps.get('fixed_state_commit',False),
        registered_context_admission=registered,
        queued_dependency_admissions=admission[0][0]if registered else 0,
        maximum_admission_wait_clocks=admission[0][1]if registered else 0,
        state_oracle=('independent rational arithmetic and input-derived admission calendar'if registered else
            'independent rational arithmetic at fixed commit'if caps.get('fixed_state_commit')else'accepted responses'),
        rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'])
    (directory/'verification_state.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build-dir',type=Path,default=ROOT/'.build/pe_12_1x1')
    check(parser.parse_args().build_dir.resolve())
