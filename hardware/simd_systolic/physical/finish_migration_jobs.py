#!/usr/bin/env python3
"""Finish the authorized physical jobs and verify their actual gate netlists.

This bounded local build step survives an app/server reconnect. It never
promotes experimental RTL or infers frequency acceptance from functional tests.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from run import ROOT,sha


def read(path):
    if not path.exists():return None
    try:return json.loads(path.read_text())
    except json.JSONDecodeError:return None  # a producer may still be writing


def finalize_execution_log(path,record):
    """Finalize a wrapper that prints its own execution record after hashing.

    Accept only that exact appended JSON record; the already hashed prefix
    must be byte-identical. Keep the producer's original record for review.
    """
    log=Path(record['execution_log']);raw=log.read_bytes()
    if hashlib.sha256(raw).hexdigest()==record['log_sha256']:return record
    trailer=(json.dumps(record,indent=2)+'\n').encode()
    if not raw.endswith(trailer)or hashlib.sha256(raw[:-len(trailer)]).hexdigest()!=record['log_sha256']:
        raise ValueError('finished producer log changed beyond its exact appended execution record')
    archived=path.with_name('bounded_execution.unfinalized.json')
    if archived.exists():raise ValueError('unfinalized execution archive already exists')
    shutil.copy2(path,archived)
    fixed=dict(record,log_sha256=sha(log),original_execution_sha256=sha(archived),
        log_hash_finalized_after_producer_exit=True)
    path.write_text(json.dumps(fixed,indent=2)+'\n')
    return fixed


def run(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if(out/'verification.json').exists():raise ValueError('completed collection already exists')
    shutil.copy2(__file__,out/'finish_driver.py')
    routed=args.routed_dir.resolve();experiment=args.experiment_out.resolve()
    for directory in(routed,experiment):directory.relative_to(ROOT)
    state=dict(kind='bounded_physical_job_collection',routed_directory=str(routed),
        experiment_directory=str(experiment),steps=[],jobs={},physical_frequency_certified=False)
    producer=None
    if args.experiment_producer_pid:
        directory=Path('/proc')/str(args.experiment_producer_pid)
        if directory.joinpath('cwd').resolve()!=ROOT:
            raise ValueError('producer cwd differs from this project')
        command=directory.joinpath('cmdline').read_bytes()
        if command.split(b'\0')[1:-1]!=[b'-u',b'-']:
            raise ValueError('expected the known local Python wrapper')
        children=[]
        for child in directory.joinpath('task',str(args.experiment_producer_pid),'children').read_text().split():
            argv=Path('/proc',child,'cmdline').read_bytes().split(b'\0')
            if b'physical/experiment_pe_context.py'in argv and str(experiment).encode()in argv:children.append(child)
        if len(children)!=1:raise ValueError('producer is not running this exact experiment')
        producer=(directory,directory.joinpath('stat').read_text().split(') ',1)[1].split()[19])
        state['experiment_producer_pid']=args.experiment_producer_pid
        state['experiment_producer_command_sha256']=hashlib.sha256(command).hexdigest()
    def save():
        temp=out/'state.json.pending';temp.write_text(json.dumps(state,indent=2)+'\n')
        temp.replace(out/'state.json')
    def call(name,command,timeout=1800):
        log=out/(name+'.log');timed_out=False
        print(name,flush=True)
        with log.open('w')as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out=True
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                code=process.wait()
        state['steps'].append(dict(name=name,command=command,returncode=code,timed_out=timed_out,
            log=str(log),log_sha256=sha(log)));save()
        return code==0 and not timed_out
    def numerical(directory,is_routed):
        identity=read(directory/'build_identity.json')
        if identity is None or identity['scope']!='int64'or identity['stages']!=2 or identity['analysis_period_ns']!=1:
            raise ValueError('collector requires an authenticated F2 I64 component at1ns')
        if sha(Path(identity['liberty_file']))!=identity['liberty_sha256']:
            raise ValueError('component process library changed')
        name='check_routed'if is_routed else'check_mapped';proof=directory/name/'verification.json'
        if proof.exists():
            record=read(proof);timing=read(directory/'timing.json')if is_routed else identity
            if not record or not record['passed']or record['rtl_sha256']!=identity['rtl_sha256']or record['netlist_sha256']!=timing['netlist_sha256'if is_routed else'mapped_sha256']:
                raise ValueError('existing gate evidence differs')
            return True
        if proof.parent.exists():raise ValueError('unfinished gate check retained for inspection: '+str(proof.parent))
        return call('routed_numerical'if is_routed else'binary_mapped_numerical',
            [sys.executable,str(ROOT/'physical/check_int64.py'),str(directory),
                '--routed'if is_routed else'--mapped','--count','1000'])
    (out/'running.json').write_text(json.dumps(dict(pid=os.getpid(),timeout_seconds=args.timeout))+'\n')
    save();deadline=time.monotonic()+args.timeout;complete=set()
    while len(complete)<2 and time.monotonic()<deadline:
        route_record=read(routed/'route_execution.json')
        if 'routed'not in complete and route_record:
            state['jobs']['routed']=dict(execution_sha256=sha(routed/'route_execution.json'),
                completed_extracted_sta=route_record['routed_sta_completed'],timed_out=route_record['timed_out'])
            if route_record['routed_sta_completed']:
                try:state['jobs']['routed']['numerical_passed']=numerical(routed,True)
                except Exception as exc:state['jobs']['routed']['numerical_error']=str(exc)
            complete.add('routed');save()
        binary_record=read(experiment/'bounded_execution.json')
        rounding=read(experiment/'rounding_sweep/verification.json')
        if producer:
            directory,start=producer
            stat=directory.joinpath('stat')
            fields=stat.read_text().split(') ',1)[1].split()if stat.exists()else[]
            if fields and fields[19]==start and fields[0]!='Z':binary_record=None
            elif binary_record:binary_record=finalize_execution_log(experiment/'bounded_execution.json',binary_record)
        if 'binary'not in complete and binary_record:
            experiment_proof=read(experiment/'verification.json')
            if experiment_proof is None:raise ValueError('finished experiment has no source verification')
            work=Path(experiment_proof['work_directory']);work.relative_to(ROOT)
            physical=work/'.build/physical/asap7_TT_int64_2_1x1_1ns_ctx_word'
            if 'binary'in state['jobs']:
                if rounding:
                    state['jobs']['binary']['rounding_sweep_sha256']=sha(experiment/'rounding_sweep/verification.json')
                    state['jobs']['binary']['rounding_passed']=rounding.get('passed',False)
                    complete.add('binary');save()
                elif(experiment/'rounding_sweep.log').exists()and'Traceback'in(experiment/'rounding_sweep.log').read_text():
                    state['jobs']['binary']['rounding_error_log_sha256']=sha(experiment/'rounding_sweep.log')
                    complete.add('binary');save()
            else:
                state['jobs']['binary']=dict(execution_sha256=sha(experiment/'bounded_execution.json'),
                    returncode=binary_record['returncode'],timed_out=binary_record['timed_out'])
                if(physical/'synth.v').exists():
                    try:state['jobs']['binary']['mapped_numerical_passed']=numerical(physical,False)
                    except Exception as exc:state['jobs']['binary']['numerical_error']=str(exc)
                if not experiment_proof.get('functional_passed'):
                    complete.add('binary')
                save()
        if len(complete)<2:time.sleep(10)
    state['all_jobs_collected']=len(complete)==2;state['timed_out']=len(complete)!=2;save()
    for name in('index_arithmetic_experiments.py','snapshot_migration.py','report_migration.py'):
        if not call(name,[sys.executable,str(ROOT/'physical'/name)]):break
    state['physical_frequency_certified']=False;save()
    (out/'verification.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--routed-dir',type=Path,required=True)
    p.add_argument('--experiment-out',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--experiment-producer-pid',type=int,help='wait for the authenticated wrapper to finish its own mapped checks before collecting')
    p.add_argument('--timeout',type=int,default=10800);args=p.parse_args()
    if not 1<=args.timeout<=14400:p.error('timeout must be within1..14400seconds')
    run(args)
