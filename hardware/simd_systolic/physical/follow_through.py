#!/usr/bin/env python3
"""Finish closed physical jobs with real routed numerical verification.

The input names already-running jobs. This does not start/restart routing or
change constraints, and a nonclosed candidate never becomes a certificate.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run import ROOT,sha
from simd_eval.certification import core_certificate


def follow(jobs_path,out,timeout,jobs):
    requested=json.loads(jobs_path.read_text())
    states=[]
    for entry in requested:
        directory=Path(entry['directory']).resolve()
        identity=json.loads((directory/'build_identity.json').read_text())
        if identity['scope']not in('pe','array','fp64'):raise ValueError('single-clock core jobs only')
        states.append(dict(directory=str(directory),parent_pid=int(entry['parent_pid']),
            rtl_sha256=identity['rtl_sha256'],stages=identity['stages'],status='waiting'))
    out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    def save():
        report=dict(scope='physical_flow_follow_through',updated_utc=datetime.now(timezone.utc).isoformat(),
            requested_jobs_sha256=sha(jobs_path),jobs=states,full_plan_complete=False,
            note='Each result belongs to its own circuit and clock. Complete-plan acceptance remains separate.')
        temporary=out/'progress.json.tmp';temporary.write_text(json.dumps(report,indent=2)+'\n')
        temporary.replace(out/'progress.json')
    save()
    while any(s['status']=='waiting'for s in states):
        if time.monotonic()-start>timeout:
            for state in states:
                if state['status']=='waiting':state['status']='host_follow_through_timeout'
            save();return
        for state in states:
            if state['status']!='waiting':continue
            directory=Path(state['directory']);timing_path=directory/'timing.json'
            if not timing_path.exists():
                failure_path=directory/'failure.json'
                if failure_path.exists():
                    try:failure=json.loads(failure_path.read_text())
                    except json.JSONDecodeError:continue
                    state.update(status='physical_flow_failed',failure_report_sha256=sha(failure_path),
                        error=failure.get('error'),routed_numerical_check_attempted=False,
                        frequency_validated=False,one_ghz_certified=False)
                    save()
                    subprocess.run([sys.executable,str(ROOT/'scripts/collect_results.py')],check=True)
                    continue
                try:os.kill(state['parent_pid'],0)
                except ProcessLookupError:
                    # A final report can be renamed just after worker exit.
                    state.setdefault('worker_missing_since',time.monotonic())
                    if time.monotonic()-state['worker_missing_since']>120:
                        state['status']='worker_exited_without_timing_report';save()
                continue
            try:timing=json.loads(timing_path.read_text())
            except json.JSONDecodeError:continue
            if timing['rtl_sha256']!=state['rtl_sha256']:raise ValueError('physical source identity changed')
            closed=all(timing.get(k)for k in('setup_hold_closed','electrical_clean','coverage_clean','route_drc_clean','constraint_policy_clean'))
            state.update(timing_report_sha256=sha(timing_path),period_ns=timing['period_ns'],
                setup_wns_ns=timing['setup_wns_ns'],hold_wns_ns=timing['hold_wns_ns'],tns_ns=timing['tns_ns'])
            if not closed:
                state.update(status='not_closed',routed_numerical_check_attempted=False,
                    frequency_validated=False,one_ghz_certified=False)
                save()
                subprocess.run([sys.executable,str(ROOT/'scripts/collect_results.py')],check=True)
                continue
            print('Closed timing; verify actual routed netlist:',directory,flush=True)
            state['status']='routed_numerical_check_running';save()
            command=[sys.executable,str(ROOT/'physical/check_gates.py'),str(directory),
                '--stages',str(state['stages']),'--routed','-j',str(jobs)]
            # Detailed routing flattens the mesh. Always verify that complete
            # circuit; synthesised PE hierarchy cannot stand in for it.
            log=directory/'follow_through_routed_gates.log'
            with log.open('w')as stream:
                result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
            state['routed_numerical_check_attempted']=True
            numeric_path=directory/'gate_verification_routed.json'
            if result.returncode or not numeric_path.exists():
                state.update(status='routed_numerical_check_failed',returncode=result.returncode,
                    frequency_validated=False,one_ghz_certified=False)
            else:
                numeric=json.loads(numeric_path.read_text())
                matched=(numeric.get('passed')and numeric.get('netlist_sha256')==timing['netlist_sha256']
                    and numeric.get('rtl_sha256')==timing['rtl_sha256'])
                state.update(status='validated'if matched else'routed_identity_mismatch',
                    frequency_validated=bool(matched),one_ghz_certified=core_certificate(timing,numeric)['passed'],
                    routed_numeric_report_sha256=sha(numeric_path))
            save()
            subprocess.run([sys.executable,str(ROOT/'scripts/collect_results.py')],check=True)
        if any(s['status']=='waiting'for s in states):time.sleep(30)
    save()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('jobs_file',type=Path)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--timeout',type=int,default=172800)
    parser.add_argument('-j',type=int,choices=[1,2,4],default=2)
    args=parser.parse_args()
    if args.timeout<1:parser.error('timeout must be positive')
    follow(args.jobs_file.resolve(),args.out.resolve(),args.timeout,args.j)
