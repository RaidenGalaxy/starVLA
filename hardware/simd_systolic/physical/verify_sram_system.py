#!/usr/bin/env python3
"""Full digital co-simulation checks for a generated SRAM implementation."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
from runtime import validated_build
from freeze_verification import verify


def run(args):
    build=args.build_dir.resolve();guard,caps=validated_build(build)
    if caps['target']!='gemm' or not caps.get('sram_native_macro'):
        raise ValueError('a GEMM build authenticated to the generated macro is required')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if(out/'acceptance.json').exists():raise ValueError('preserve existing verification results; choose another --out')
    driver=Path(__file__).read_bytes();driver_sha=hashlib.sha256(driver).hexdigest()
    (out/'verification_driver.py').write_bytes(driver)
    cohort=verify(args.cohort)
    if caps.get('verification_cohort_sha256')!=cohort['cohort_sha256']:raise ValueError('source cohort differs from the model build')
    report=dict(kind='generated_sram_gemm_digital_verification',rtl_sha256=caps['rtl_sha256'],
        simulator_sha256=caps['simulator_sha256'],testbench_sha256=caps['testbench_sha256'],
        macro_package_manifest_sha256=caps['sram_package_manifest_sha256'],
        native_macro=caps['sram_native_macro'],source_cohort_sha256=cohort['cohort_sha256'],
        driver_sha256=driver_sha,steps=[],functional_model_passed=False,
        physical_frequency_certified=False,transistor_behavior_verified=False)
    def save():(out/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    def call(name,arguments,overrides=None):
        verify(args.cohort)
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=driver_sha:raise ValueError('verification driver changed')
        log=out/(name+'.log');print(name,flush=True)
        with log.open('w')as stream:
            result=subprocess.run([sys.executable,*map(str,arguments)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                env={**os.environ,**(overrides or {})})
        report['steps'].append(dict(name=name,command=list(map(str,arguments)),returncode=result.returncode,
            environment_overrides=overrides or {},log=str(log),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
        save()
        if result.returncode:raise RuntimeError(name+' failed: '+str(log))
        verify(args.cohort)
    save();rows,cols=caps['rows'],caps['cols']
    for script in ('check_descriptor','check_grouped','check_epilogue'):
        shape=['--rows',rows,'--cols',cols]if script!='check_epilogue'else[]
        call(script,['scripts/'+script+'.py','--build-dir',build,'--dram','--out',out/script,*shape])
    call('prefetch',['scripts/check_native_prefetch.py','--build-dir',build,'--dram','--out',out/'prefetch'])
    call('scratch',['scripts/check_scratch.py','--build-dir',build,'--full','--out',out/'scratch'])
    for domain in ('both','core','memory'):
        call('reset_'+domain,['scripts/check_reset.py','--build-dir',build,'--dram','--out',out/('reset_'+domain)],
             {'SIMD_RESET_DOMAIN':domain})
    call('faults',['scripts/check_faults.py','--build-dir',build,'--dram','--out',out/'faults'])
    call('cdc',['scripts/check_system_cdc.py','--build-dir',build])
    for phase in (0,250,500,1500):
        call('phase'+str(phase),['scripts/check_descriptor.py','--build-dir',build,'--rows',rows,'--cols',cols,
            '--dram','--smoke','--out',out/('phase'+str(phase))],{'SIMD_MEMORY_PHASE_PS':str(phase),'SIMD_CORE_PHASE_PS':'0'})
    hashes=[]
    for phase in (0,250,500,1500):
        directory=out/('phase'+str(phase))
        hashes.append(json.loads((directory/'metrics.json').read_text())['output_sha256'])
        observed=[list(map(int,line.split()[1:]))for line in (directory/'simulation.log').read_text().splitlines()if line.startswith('J ')]
        if observed!=[[1000,2000,1000,0,phase]]:raise ValueError('runtime clock configuration differs from the target')
    if len(set(hashes))!=1:raise ValueError('clock phase changes output bytes')
    report.update(functional_model_passed=True,clock_phase_output_equal=True);save()
    print(json.dumps({k:report[k]for k in ('functional_model_passed','physical_frequency_certified','transistor_behavior_verified')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--cohort',type=Path,required=True)
    run(p.parse_args())
