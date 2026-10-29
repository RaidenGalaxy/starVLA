#!/usr/bin/env python3
"""Flush a full response queue and a queued dependency, then check all formats."""
import argparse
import json
from pathlib import Path
import subprocess
from check_pe import make_vectors,verify_output,ROOT
from runtime import validated_build


def verify_reset_coverage(stdout,caps):
    resets=[list(map(int,line.split()[1:]))for line in stdout.splitlines()if line.startswith('Z ')]
    expected=[[0,caps['pe_response_credits']],[1,2]]if caps.get('registered_context_admission')else[[0,2],[1,1]]
    if resets!=expected:
        raise AssertionError('missing full-response and buffered-dependency reset coverage')
    registered=caps.get('registered_context_admission',False)
    return dict(resets=2,reset_storage_kind='reserved_response_queue'if registered else'elastic_pipeline',
                reserved_response_entries_flushed=resets[0][1]if registered else 0,
                elastic_pipeline_entries_flushed=0 if registered else resets[0][1],
                uncommitted_dependent_requests_flushed=resets[1][1]if registered else 0,
                uncommitted_requests_flushed=resets[1][1],
                all_contexts_zero_after_reset=True,no_ghost_responses=True)


def check(directory):
    guard,caps=validated_build(directory)
    if caps['target']!='pe':raise ValueError('PE build required')
    commands,expected,acc,sticky=make_vectors()
    trace=directory/'reset_vectors.txt';trace.write_text(''.join(commands))
    proc=subprocess.run([str(directory/'sim'),str(trace),'--reset-stress'],capture_output=True,text=True,timeout=300)
    (directory/'reset_simulation.log').write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError(f'PE reset check failed; see {directory/"reset_simulation.log"}')
    coverage=verify_reset_coverage(proc.stdout,caps)
    count=verify_output(proc.stdout,expected,acc,sticky,caps['state_commit_latency'])
    report=dict(passed=True,scope='pe_reset_recovery',stages=caps['stages'],post_reset_vectors=count,
                **coverage,rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'])
    (directory/'verification_reset.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build-dir',type=Path,
        default=ROOT/'.build/pe_12_1x1')
    check(parser.parse_args().build_dir.resolve())
