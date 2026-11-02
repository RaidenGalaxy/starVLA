#!/usr/bin/env python3
"""Replay all-format and reset tests on the actual corrected system gate netlist.

Zero-delay gate behavior is checked against previously qualified independent
reference outputs. SRAM timing and analog behavior require separate evidence.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from run import ROOT,sha
from asap7_platform import profile
from async_mapping_policy import validate_checkpoint
from format_mapped_verilog import wrap_assignments

sys.path.insert(0,str(ROOT/'scripts'))
from runtime import validated_build


def written(text):
    result={}
    for line in text.splitlines():
        if line.startswith('W '):
            _,address,value=line.split();result[int(address)]=int(value)
    return result


def run(args):
    source=args.source.resolve();reference=args.reference_build.resolve()
    proof=source/'verification.json';prior=json.loads(proof.read_text())
    validate_checkpoint(source,prior)
    for filename,digest in prior['evidence'].items():
        if sha(filename)!=digest:raise ValueError('corrected system source changed: '+filename)
    guard,caps=validated_build(reference)
    if(caps['target'],caps['stages'],caps['rows'],caps['cols'])!=('gemm',2,2,2)or caps['rtl_sha256']!=prior['source_rtl_sha256']:
        raise ValueError('requires the matching qualified native2x2 RTL reference')
    out=args.out.resolve()
    if out.exists():raise ValueError('preserve system gate-check attempts')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'gate_check_driver.py')
    state=dict(kind='actual_corrected_system_gate_regression',passed=False,arithmetic_stages=2,
        core_period_ps=1000,memory_period_ps=2000,dram_period_ps=1000,
        zero_delay_gate_simulation=True,physical_frequency_certified=False,
        sram_transistor_behavior_verified=False,source_netlist_sha256=sha(source/'repaired.v'),
        reference_rtl_sha256=caps['rtl_sha256'],reference_simulator_sha256=caps['simulator_sha256'],
        source_correction_verification=str(proof),source_correction_verification_sha256=sha(proof),steps=[],cases=[])
    def save():
        pending=out/'state.json.pending';pending.write_text(json.dumps(state,indent=2)+'\n');pending.replace(out/'state.json')
    def call(name,command,timeout,extra_env=None):
        log=out/(name+'.log');state['active_phase']=name;save();timed_out=False
        with log.open('w')as stream:
            process=subprocess.Popen([str(x)for x in command],stdout=stream,stderr=subprocess.STDOUT,
                cwd=ROOT,env={**os.environ,**(extra_env or {})},start_new_session=True)
            state['active_pid']=process.pid;save()
            try:code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out=True
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                code=process.wait()
        state.pop('active_pid',None)
        state['steps'].append(dict(name=name,command=[str(x)for x in command],returncode=code,
            timed_out=timed_out,timeout_seconds=timeout,environment=extra_env or {},log=str(log),log_sha256=sha(log)))
        save()
        if code or timed_out:raise RuntimeError('system gate phase failed: '+name)
        return log.read_text()
    try:
        bench=out/'testbench';shutil.copytree(reference/'testbench',bench)
        for name,digest in caps['testbench_manifest'].items():
            if sha(bench/name)!=digest:raise ValueError('reference testbench snapshot changed: '+name)
        shutil.copytree(reference/'libraries',out/'libraries')
        library=profile('TT')['liberty'];cells=out/'cells.v'
        call('cell_models',['yosys','-Q','-T','-p',f'read_liberty -ignore_miss_func {library}; write_verilog {cells}'],120)
        wrapper=reference/'inputs/Asap7SramMacro.sv'
        shutil.copy2(wrapper,out/'sram_model.sv')
        mapped=out/'tool_input.v';shutil.copy2(source/'repaired.v',mapped)
        shutil.copy2(ROOT/'physical/format_mapped_verilog.py',out/'netlist_formatter.py')
        state['wrapped_assignments']=wrap_assignments(mapped)
        state['simulation_netlist_sha256']=sha(mapped)
        flags=['-CFLAGS','-O1 -std=c++17','-LDFLAGS',f'-L{out/"libraries"} -Wl,-rpath,{out/"libraries"} -ldramsim3']
        build_log=call('build',['verilator','--cc','--exe','--build','-j',str(args.jobs),'--assert','-Wno-fatal',
            '--top-module','GemmSystem','--Mdir',out/'obj','--output-split','20000',
            *flags,cells,out/'sram_model.sv',mapped,bench/'gemm_main.cpp','-o',out/'sim'],args.build_timeout)
        if '%Warning-MULTIDRIVEN:'in build_log:raise ValueError('gate export has multiple drivers; do not qualify it')
        missing=re.findall(r"Instance has missing pin: '([^']+)'",build_log)
        if set(missing)-{'Y'}:raise ValueError('gate export has unexpected disconnected pins')
        state['unconnected_clock_load_outputs']=len(missing)
        state['simulator_sha256']=sha(out/'sim');save()
        config=ROOT/'configs/ddr4.ini';qualified=ROOT/'artifacts/migration_asap7_f2/generated_sram_gemm'
        for phase in(0,250,500,1500):
            directory=qualified/('phase'+str(phase));metrics=json.loads((directory/'metrics.json').read_text())
            trace=directory/'commands.txt'
            if(not metrics['passed']or not metrics['exact_output_match']or metrics['descriptors']!=29
                    or metrics['rtl_sha256']!=caps['rtl_sha256']or sha(trace)!=metrics['trace_sha256']
                    or sha(config)!=metrics['dram_config_sha256']):
                raise ValueError('all-format independent-reference evidence differs')
            golden=written((directory/'simulation.log').read_text())
            actual=call('phase'+str(phase),[out/'sim',trace,'1000','2000',config,out/('dram_phase'+str(phase)),'207'],
                args.case_timeout,{'SIMD_MEMORY_PHASE_PS':str(phase),'SIMD_CORE_PHASE_PS':'0'})
            if written(actual)!=golden:raise ValueError('all-format gate output differs from the qualified reference')
            if[f for f in actual.splitlines()if f.startswith('J ')]!=[f'J 1000 2000 1000 0 {phase}']:
                raise ValueError('gate runtime clock configuration differs')
            completed=[line for line in actual.splitlines()if line.startswith('D ')]
            if len(completed)!=29:raise ValueError('gate descriptor completion count differs')
            state['cases'].append(dict(name='phase'+str(phase),passed=True,descriptors=29,formats=11,
                verified_output_bytes=len(golden),trace=str(trace),trace_sha256=sha(trace),
                qualified_metrics=str(directory/'metrics.json'),qualified_metrics_sha256=sha(directory/'metrics.json'),
                reference_log_sha256=sha(directory/'simulation.log')));save()
        for domain in('both','core','memory'):
            for phase in('r','c','w'):
                trace=qualified/('reset_'+domain)/(phase+'.txt')
                actual=call('reset_'+domain+'_'+phase,[out/'sim',trace,'1000','2000',config,
                    out/('dram_reset_'+domain+'_'+phase),'207'],args.case_timeout,
                    {'SIMD_RESET_DOMAIN':domain,'SIMD_CORE_PHASE_PS':'0','SIMD_MEMORY_PHASE_PS':'0'})
                done=[list(map(int,line.split()[1:]))for line in actual.splitlines()if line.startswith('D ')]
                observed=written(actual);new={a:b for a,b in observed.items()if a>=0x90000000}
                expected=dict(zip(range(0x90000003,0x90000007),(-5&0xffffffff).to_bytes(4,'little')))
                resets=[line.split()[1:]for line in actual.splitlines()if line.startswith('XR ')]
                if(len(done)!=1 or done[0][0]!=18 or done[0][1]!=0 or done[0][3:5]!=[3,0]or new!=expected
                        or resets!=[[domain,str(int(domain!='memory')),str(int(domain!='core'))]]):
                    raise ValueError('reset/restart gate arithmetic or domain mismatch')
                if phase!='w'and any(0x80000000<=a<0x90000000 for a in observed):
                    raise ValueError('pre-write gate reset wrote an aborted descriptor')
                state['cases'].append(dict(name='reset_'+domain+'_'+phase,passed=True,
                    trace=str(trace),trace_sha256=sha(trace),expected_restarted_result=-5));save()
        state['passed']=True
    except Exception as error:
        state['error']=str(error);raise
    finally:
        state.pop('active_phase',None);state.pop('active_pid',None)
        state['evidence']={str(p):sha(p)for p in [proof,source/'repaired.v',reference/'capabilities.json',
            *[p for p in out.iterdir()if p.is_file()and p.name not in('state.json','verification.json')]]}
        save();(out/'verification.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reference-build',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=4);p.add_argument('--build-timeout',type=int,default=7200)
    p.add_argument('--case-timeout',type=int,default=1800);a=p.parse_args()
    if not 1<=a.jobs<=8 or not 1<=a.build_timeout<=7200 or not 1<=a.case_timeout<=3600:p.error('jobs1..8,build1..7200s,case1..3600s')
    run(a)
