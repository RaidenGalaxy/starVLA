#!/usr/bin/env python3
"""Reproducible F2/ASAP7 functional acceptance, separate from physical signoff."""
import argparse
import concurrent.futures
import threading
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from simd_eval.architecture import ARCH
from freeze_verification import freeze,verify
from audit_sram import audit


def run(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    cohort_path=out/'cohort.json'
    cohort=verify(cohort_path)if cohort_path.exists()else freeze(cohort_path)
    label='migration_'+cohort['cohort_sha256'][:12]
    report=dict(architecture=ARCH,cohort_sha256=cohort['cohort_sha256'],steps=[],
                functional_passed=False,full_hardware_plan_complete=False,
                physical_frequency_certified=False,sram_audit=audit())
    report_lock=threading.RLock()
    def save():
        with report_lock:
            pending=out/'acceptance.json.tmp'
            pending.write_text(json.dumps(report,indent=2)+'\n');pending.replace(out/'acceptance.json')
    def call(name,arguments,env=None):
        verify(cohort_path)
        print(name,flush=True);log=out/(name+'.log')
        with log.open('w')as stream:
            process=subprocess.run([sys.executable,*map(str,arguments)],cwd=ROOT,
                stdout=stream,stderr=subprocess.STDOUT,env=env)
        record=dict(name=name,command=list(map(str,arguments)),returncode=process.returncode,
                    log=str(log),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
        record['environment_overrides']={k:v for k,v in (env or {}).items()if k in ('SIMD_RESET_DOMAIN','SIMD_MEMORY_PHASE_PS','SIMD_CORE_PHASE_PS')}
        with report_lock:
            report['steps'].append(record);save()
        if process.returncode:raise RuntimeError(name+' failed; inspect '+str(log))
        verify(cohort_path)
    configurations=[('alu',1),('pe',1),('array',8),('system',2),('gemm',2),('gemm',8)]
    builds={}
    for kind,rows in configurations:
        flags=['--hierarchical']if kind!='alu'else[]
        if rows==8:flags+=['--sim-threads',4]
        call('build_'+kind+str(rows),['scripts/build.py','--kind',kind,'--stages',2,'--rows',rows,
            '--label',label,'--cohort',cohort_path,'-j',args.jobs,*flags])
        builds[kind,rows]=ROOT/f'.build/{kind}_2_{rows}x{rows}_{label}{"_mt4" if rows==8 else ""}' 
    call('unit_tests',['-m','unittest','discover','-s','tests','-p','test_*.py','-v'])
    call('sram_bank',['scripts/check_sram_bank.py'])
    for stall in (False,True):
        call('alu_stall'if stall else'alu',['scripts/check_alu.py','--build-dir',builds['alu',1],
            '--exhaustive',*(['--stall']if stall else[])])
    call('alu_idle_resume',['scripts/check_alu.py','--build-dir',builds['alu',1],'--exhaustive','--idle-gaps','--stall'])
    for name in ('check_pe','check_pe_state','check_ii'):
        call(name,['scripts/'+name+'.py','--build-dir',builds['pe',1]])
    for fmt in ('I32','I64'):
        for k in (1,7,8,9,65535,65536):
            call('long_'+fmt+'_'+str(k),['scripts/check_pe_long_k.py','--stages',2,'--build-dir',builds['pe',1],'--format',fmt,'--k',k])
    call('f2_reference',['scripts/check_equivalence.py','--stages',2,'--build-dirs',builds['alu',1],'--out',out/'reference'])
    for name in ('check_array','check_array_events'):
        dimensions=['--stages',2,'--rows',8,'--cols',8]if name=='check_array'else[]
        call(name,['scripts/'+name+'.py','--build-dir',builds['array',8],'--pipelined-read',*dimensions])
    call('array_cycles',['scripts/check_array_cycles.py','--build-dir',builds['array',8]])
    call('array_hazards',['scripts/check_array_hazard.py','--build-dir',builds['array',8]])
    call('auxiliary',['scripts/check_auxiliary.py','--build-dir',builds['system',2],'--dram'])
    call('dram_adapter',['scripts/check_dram_adapter.py'])
    def system_suite(rows):
        build=builds['gemm',rows]
        for script in ('check_descriptor','check_grouped','check_epilogue'):
            dimension=['--rows',rows,'--cols',rows]if script!='check_epilogue'else[]
            call(script+str(rows),['scripts/'+script+'.py','--build-dir',build,'--dram',
                '--out',out/(script+str(rows)),*dimension])
        call('prefetch'+str(rows),['scripts/check_native_prefetch.py','--build-dir',build,'--dram','--out',out/('prefetch'+str(rows))])
        call('scratch'+str(rows),['scripts/check_scratch.py','--build-dir',build,'--full','--out',out/('scratch'+str(rows))])
        for domain in ('both','core','memory'):
            call('reset'+str(rows)+'_'+domain,['scripts/check_reset.py','--build-dir',build,'--dram',
                '--out',out/('reset'+str(rows)+'_'+domain)],env={**os.environ,'SIMD_RESET_DOMAIN':domain})
        call('faults'+str(rows),['scripts/check_faults.py','--build-dir',build,'--dram','--out',out/('faults'+str(rows))])
        call('cdc'+str(rows),['scripts/check_system_cdc.py','--build-dir',build])
    def phase_suite():
        for phase in (0,250,500,1500):
            call('phase'+str(phase),['scripts/check_descriptor.py','--build-dir',builds['gemm',8],
                '--rows',8,'--cols',8,'--dram','--smoke','--out',out/('phase'+str(phase))],
                env={**os.environ,'SIMD_MEMORY_PHASE_PS':str(phase)})
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_suites) as pool:
        futures=[pool.submit(system_suite,2),pool.submit(system_suite,8),pool.submit(phase_suite)]
        for future in concurrent.futures.as_completed(futures):future.result()
    digests=[json.loads((out/('phase'+str(p))/'metrics.json').read_text())['output_sha256']for p in (0,250,500,1500)]
    if len(set(digests))!=1:raise AssertionError('clock phase changed output bytes')
    for phase in (0,250,500,1500):
        observed=[list(map(int,line.split()[1:]))for line in (out/('phase'+str(phase))/'simulation.log').read_text().splitlines()if line.startswith('J ')]
        if observed!=[[1000,2000,1000,0,phase]]:raise AssertionError('runtime clock periods/phase differ from the requested case')
    report.update(functional_passed=True,clock_phase_output_equal=True,builds={str(k):str(v)for k,v in builds.items()})
    save();print(json.dumps({k:report[k]for k in ('functional_passed','full_hardware_plan_complete','physical_frequency_certified')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=int,default=8)
    parser.add_argument('--parallel-suites',type=int,choices=[1,2,3],default=2)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/migration_asap7_f2/functional')
    run(parser.parse_args())
