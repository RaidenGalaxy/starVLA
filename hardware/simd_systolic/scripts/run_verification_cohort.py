#!/usr/bin/env python3
"""One frozen-source verification campaign, with resumable honest acceptance.

Every configuration has a pre-run model. Numerical subsets cannot certify a
configuration until all mandatory protocol/system/clock scenarios also pass.
No physical frequency or whole hardware-plan acceptance is inferred here.
"""
import argparse
import concurrent.futures
import configparser
import datetime
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
import sys as _architecture_sys
_architecture_sys.path.insert(0,str(ROOT/'python'))
from simd_eval.architecture import CORE_PS, MEMORY_PS
sys.path.insert(0,str(ROOT/'python'))
from freeze_verification import freeze,verify,sha
from runtime import validated_build
from simd_eval.reference import Format
from simd_eval.functional_coverage import FunctionalCoverage,numerical_bins,bin_spec


def required_scenarios(kind):
    if kind=='pe':
        names=['random_all_formats','live_state','reset','ii1','alu_exhaustive','code_differential']
    elif kind=='array':
        names=['random_matrix','cycle_perturbation','boundary_events','hazard','random_cycle_stress','code_differential']
    else:
        return ['native_dram','native_ideal','ideal_dram_equal','phase500','epilogue','faults','reset',
                'structural_cdc','fifo_phases','scratch','burst32','burst128','code_differential']+[
                    'group_'+mode for mode in ('W4A4','W4A8','W8A8','FP4','FP8')]
    return names+[f'long_{fmt}_{k}'for fmt in('I32','I64')for k in(1,7,8,9,65535,65536)]


def system_bins():
    bins=[]
    for fmt in Format:
        for mode in('zero','initial'):
            bins.append(bin_spec(f'native/{fmt.name}/{mode}','native_gemm_formats'))
        for dst in Format:bins.append(bin_spec(f'epilogue/{fmt.name}/{dst.name}','input_output_formats'))
    for shape in ('1x1x1','7x19x65'):bins.append(bin_spec('shape/'+shape,'matrix_shapes'))
    for fmt in ('I4','I8','F4','F8'):bins.append(bin_spec('grouped/'+fmt,'group_boundaries'))
    return bins


class Campaign:
    def __init__(self,args):
        self.args=args;self.out=args.out.resolve();self.out.mkdir(parents=True,exist_ok=True)
        self.cohort_path=self.out/'cohort.json'
        self.cohort=verify(self.cohort_path)if self.cohort_path.exists()else freeze(self.cohort_path)
        self.label='cohort_'+self.cohort['cohort_sha256'][:12]
        self.lock=threading.Lock();self.progress={}
        old=self.out/'progress.json'
        if old.exists():self.progress=json.loads(old.read_text())['configurations']

    def update(self,key,**fields):
        with self.lock:
            self.progress.setdefault(key,{}).update(fields)
            report=dict(cohort_sha256=self.cohort['cohort_sha256'],updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        configurations=self.progress,verification_campaign_complete=False,
                        full_hardware_plan_complete=False,one_ghz_certified=False)
            path=self.out/'progress.json';pending=path.with_suffix('.json.tmp')
            pending.write_text(json.dumps(report,indent=2)+'\n');pending.replace(path)

    def configuration(self,kind,stages,rows):
        key=f'{kind}_{stages}_{rows}x{rows}'
        if self.progress.get(key,{}).get('status')=='passed':
            record=self.progress[key]
            if sha(record['report'])!=record['report_sha256']:raise ValueError('completed report changed')
            return
        root=self.out/key;root.mkdir(exist_ok=True)
        attempt=root/f'attempt_{len(list(root.glob("attempt_*")))+1}'
        attempt.mkdir();step=0
        def call(name,script,*arguments,env=None):
            nonlocal step
            verify(self.cohort_path);step+=1;self.update(key,status='running',step=name,attempt=str(attempt))
            log=attempt/f'{step:02d}_{name}.log';command=[sys.executable,str(ROOT/script),*map(str,arguments)]
            metadata=dict(command=command,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            with log.open('w')as stream:
                run=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env)
            metadata.update(returncode=run.returncode,log_sha256=sha(log),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            log.with_suffix('.step.json').write_text(json.dumps(metadata,indent=2)+'\n')
            if run.returncode:raise RuntimeError(f'{key}: {name} failed; {log}')
            verify(self.cohort_path)
        try:
            prefix=ROOT/f'.build/{kind}_{stages}_{rows}x{rows}_{self.label}'
            threads=4 if kind in('array','gemm')else 1
            thread_suffix='_mt4'if threads==4 else''
            base=Path(str(prefix)+thread_suffix)
            common=['--kind',kind,'--stages',stages,'--rows',rows,'--label',self.label,
                    '--cohort',self.cohort_path,'-j',self.args.jobs]
            if kind!='alu':common+=['--hierarchical']
            if threads==4:common+=['--sim-threads',4]
            call('build','scripts/build.py',*common)
            guard,caps=validated_build(base)
            identity={k:caps[k]for k in('target','stages','rows','cols','rtl_sha256','simulator_sha256','verification_cohort_sha256')}
            bins=system_bins()if kind=='gemm'else numerical_bins(kind,rows,rows)
            bins+= [bin_spec('suite/'+name,'required_scenarios')for name in required_scenarios(kind)]
            coverage=FunctionalCoverage(identity,bins)
            (attempt/'configuration_model.json').write_text(json.dumps(coverage.model,indent=2)+'\n')
            def hit(name,path):
                proof=json.loads(Path(path).read_text())
                if not proof.get('passed'):raise ValueError('required check did not pass: '+str(path))
                for field in ('rtl_sha256','simulator_sha256'):
                    if field in proof and proof[field]!=caps[field]:raise ValueError('cross-build required evidence')
                coverage.checked(['suite/'+name],witness=dict(report=str(path),sha256=sha(path)),actual=True,expected=True)
                checkpoint()
                return proof
            def checkpoint():
                report=coverage.report();report['full_hardware_plan_complete']=False
                (attempt/'functional_coverage.json').write_text(json.dumps(report,indent=2)+'\n')
            if kind in('pe','array'):
                call('numerical','scripts/check_functional_coverage.py','--build-dir',base,'--out',attempt/'numerical','--timeout',43200)
                numeric=json.loads((attempt/'numerical/functional_coverage.json').read_text())
                if not numeric['passed']or numeric['identity']['rtl_sha256']!=caps['rtl_sha256']:raise ValueError('invalid numerical evidence')
                for name,count in numeric['hits'].items():
                    coverage.checked([name],witness=numeric['first_witnesses'][name],actual=True,expected=True)
                checkpoint()
                if kind=='pe':
                    alu=ROOT/f'.build/alu_{stages}_1x1_{self.label}'
                    call('alu_build','scripts/build.py','--kind','alu','--stages',stages,'--label',self.label,
                         '--cohort',self.cohort_path,'-j',self.args.jobs)
                    call('alu_exhaustive','scripts/check_alu.py','--build-dir',alu,'--stages',stages,'--count',200,'--stall','--exhaustive')
                    ag,ac=validated_build(alu);proof=alu/'verification_stall.json';ap=json.loads(proof.read_text())
                    if not ap['passed']or ap['rtl_sha256']!=ac['rtl_sha256']or ap['simulator_sha256']!=ac['simulator_sha256']:
                        raise AssertionError('ALU exhaustive evidence identity mismatch')
                    coverage.checked(['suite/alu_exhaustive'],witness=dict(report=str(proof),sha256=sha(proof),component='same_cohort_ALU'),actual=True,expected=True)
                    for name,script,report in [('random_all_formats','check_pe.py','verification.json'),
                        ('live_state','check_pe_state.py','verification_state.json'),
                        ('reset','check_pe_reset.py','verification_reset.json'),('ii1','check_ii.py','verification_ii.json')]:
                        arguments=['--build-dir',base]
                        if name in('random_all_formats','ii1'):arguments+=['--stages',stages]
                        call(name,'scripts/'+script,*arguments);hit(name,base/report)
                else:
                    call('random_matrix','scripts/check_array.py','--build-dir',base,'--stages',stages,'--rows',rows,'--cols',rows,'--pipelined-read')
                    hit('random_matrix',base/'verification.json')
                    call('cycle_perturbation','scripts/check_array_cycles.py','--build-dir',base)
                    hit('cycle_perturbation',base/'verification_cycles.json')
                    call('boundary_events','scripts/check_array_events.py','--build-dir',base,'--pipelined-read','--timeout',43200)
                    hit('boundary_events',base/'verification_events.json')
                    call('hazard','scripts/check_array_hazard.py','--build-dir',base);hit('hazard',base/'verification_hazard.json')
                    call('random_cycle_stress','scripts/check_cycle_stress.py','--build-dir',base,'--out',attempt/'cycle_stress')
                    hit('random_cycle_stress',attempt/'cycle_stress/verification.json')
                for fmt in('I32','I64'):
                    script='check_pe_long_k.py'if kind=='pe'else'check_long_k.py'
                    arguments=['--build-dir',base,'--stages',stages,'--format',fmt,'--k',65536,'--timeout',43200]
                    if kind=='array':arguments+=['--checkpoints',1,7,8,9,65535,65536]
                    call('long_'+fmt,'scripts/'+script,*arguments)
                    filename=f'verification_long_k_pe_{fmt}_65536.json'if kind=='pe'else f'verification_long_k_{fmt}_65536.json'
                    evidence=json.loads((base/filename).read_text())
                    for k in(1,7,8,9,65535,65536):
                        if kind=='pe':assert evidence['verified_intermediate_results']>=k
                        else:assert k in evidence['verified_checkpoints']
                        hit(f'long_{fmt}_{k}',base/filename)
            else:
                core=CORE_PS
                def native(name,dram=True,memory=MEMORY_PS,smoke=False,build=base,env=None):
                    arguments=['--build-dir',build,'--stages',stages,'--rows',rows,'--cols',rows,
                               '--core-ps',core,'--memory-ps',memory,'--out',attempt/name,'--timeout',43200]
                    if dram:arguments+=['--dram']
                    if smoke:arguments+=['--smoke']
                    call(name,'scripts/check_descriptor.py',*arguments,env=env)
                    return json.loads((attempt/name/'metrics.json').read_text())
                real=native('native_dram');hit('native_dram',attempt/'native_dram/metrics.json')
                if not real.get('dram_runtime_geometry_observed')or not real.get('runtime_port_monitor_observed'):
                    raise ValueError('missing runtime DRAM/port observations')
                jobs=json.loads((attempt/'native_dram/jobs.json').read_text())
                for tag,q in jobs.items():
                    ids=[f'native/{q["fmt"]}/{"initial"if q["accumulate"]else"zero"}']
                    shape=f'{q["m"]}x{q["n"]}x{q["k"]}'
                    if shape in('1x1x1','7x19x65'):ids+=['shape/'+shape]
                    coverage.checked(ids,witness=dict(done_tag=tag,source=str(attempt/'native_dram/simulation.log')),actual=True,expected=True)
                ideal=native('native_ideal',False);hit('native_ideal',attempt/'native_ideal/metrics.json')
                if real['output_sha256']!=ideal['output_sha256']:raise AssertionError('ideal and DRAM result bytes differ')
                coverage.checked(['suite/ideal_dram_equal'],witness=dict(real=real['output_sha256'],ideal=ideal['output_sha256']),actual=real['output_sha256'],expected=ideal['output_sha256'])
                native('phase500',True,MEMORY_PS,env={**os.environ,'SIMD_MEMORY_PHASE_PS':'500'});hit('phase500',attempt/'phase500/metrics.json')
                call('epilogue','scripts/check_epilogue.py','--build-dir',base,'--out',attempt/'epilogue','--dram','--core-ps',core,'--memory-ps',MEMORY_PS,'--timeout',43200)
                hit('epilogue',attempt/'epilogue/verification.json')
                for tag,q in json.loads((attempt/'epilogue/jobs.json').read_text()).items():
                    ids=[f'epilogue/{q["fmt"]}/{q["output_format"]}']
                    if q['grouped']:ids+=['grouped/'+q['fmt']]
                    coverage.checked(ids,witness=dict(done_tag=tag,source=str(attempt/'epilogue/simulation.log')),actual=True,expected=True)
                for mode in('W4A4','W4A8','W8A8','FP4','FP8'):
                    name='group_'+mode
                    call(name,'scripts/check_grouped.py','--build-dir',base,'--stages',stages,'--rows',rows,'--cols',rows,
                         '--mode',mode,'--dram','--core-ps',core,'--memory-ps',MEMORY_PS,'--out',attempt/name)
                    hit(name,attempt/name/'metrics.json')
                for name,script in [('faults','check_faults.py'),('reset','check_reset.py')]:
                    call(name,'scripts/'+script,'--build-dir',base,'--stages',stages,'--dram','--out',attempt/name)
                    hit(name,attempt/name/'verification.json')
                call('structural_cdc','scripts/check_system_cdc.py','--build-dir',base)
                hit('structural_cdc',ROOT/'.build'/('system_cdc_'+base.name)/'verification.json')
                fifo_path=self.out/'global/cdc_matrix/verification.json';fifo=json.loads(fifo_path.read_text())
                if not fifo['passed']or fifo['rtl_sha256']!=sha(base/'inputs/AsyncFifo.sv'):raise AssertionError('FIFO matrix source differs from this system')
                coverage.checked(['suite/fifo_phases'],witness=dict(report=str(fifo_path),sha256=sha(fifo_path)),actual=True,expected=True)
                call('scratch','scripts/check_scratch.py','--build-dir',base,'--full','--out',attempt/'scratch')
                hit('scratch',attempt/'scratch/verification.json')
                for burst in(32,128):
                    config=configparser.ConfigParser();config.read(ROOT/'configs/ddr4.ini');config['system']['bus_width']=str(burst)
                    path=attempt/f'burst{burst}.ini'
                    with path.open('w')as stream:config.write(stream)
                    call(f'burst{burst}','scripts/check_epilogue.py','--build-dir',base,'--out',attempt/f'burst{burst}',
                         '--dram','--dram-config',path,'--smoke','--core-ps',core,'--memory-ps',MEMORY_PS,'--timeout',43200)
                    hit(f'burst{burst}',attempt/f'burst{burst}/verification.json')
            # Coverage changes the host model only. Re-run identical directed
            # inputs and compare actual output/cycle logs before reporting it.
            call('coverage_build','scripts/build.py',*common,'--coverage')
            covered=Path(str(prefix)+'_coverage'+thread_suffix);cg,cc=validated_build(covered)
            if cc['rtl_sha256']!=caps['rtl_sha256']:raise AssertionError('coverage altered RTL identity')
            if kind in('pe','array'):
                call('code_run','scripts/check_functional_coverage.py','--build-dir',covered,'--out',attempt/'code','--timeout',43200)
                normal_log=attempt/'numerical/simulation.log';code_log=attempt/'code/simulation.log'
            else:
                normal=native('code_reference',smoke=True)
                env=os.environ.copy();env['SIMD_CODE_COVERAGE']=str(attempt/'code/coverage.dat');(attempt/'code').mkdir()
                instrumented=native('code',smoke=True,build=covered,env=env)
                if normal['output_sha256']!=instrumented['output_sha256']:raise AssertionError('coverage changed system result bytes')
                normal_log=attempt/'code_reference/simulation.log';code_log=attempt/'code/simulation.log'
                data=attempt/'code/coverage.dat'
                data.with_name(data.name+'.identity.json').write_text(json.dumps(dict(rtl_sha256=cc['rtl_sha256'],simulator_sha256=cc['simulator_sha256'],coverage_sha256=sha(data),functional_run_passed=True),indent=2)+'\n')
            if sha(normal_log)!=sha(code_log):raise AssertionError('coverage changed observed numerical/protocol/cycle log')
            call('code_report','scripts/report_code_coverage.py',attempt/'code/coverage.dat','--build-dir',covered,'--out',attempt/'code_coverage.json')
            coverage.checked(['suite/code_differential'],witness=dict(normal_log_sha256=sha(normal_log),coverage_log_sha256=sha(code_log)),actual=True,expected=True)
            checkpoint();report=coverage.report();report.update(verification_cohort_sha256=self.cohort['cohort_sha256'],
                full_hardware_plan_complete=False,physical_signoff=False,code_coverage_report=str(attempt/'code_coverage.json'))
            path=attempt/'functional_coverage.json';path.write_text(json.dumps(report,indent=2)+'\n')
            if not report['passed']:raise AssertionError('unhit mandatory functional bins')
            self.update(key,status='passed',step='complete',report=str(path),report_sha256=sha(path),functional_coverage_percent=report['functional_coverage_percent'])
        except Exception as error:
            self.update(key,status='failed',error=str(error));raise

    def run(self):
        global_dir=self.out/'global';global_dir.mkdir(exist_ok=True)
        for name,command in [('unit',[sys.executable,'-m','unittest','discover','-s','tests','-p','test_*.py']),
                             ('cdc',[sys.executable,'scripts/check_cdc_matrix.py','--out',str(global_dir/'cdc_matrix'),'-j',str(self.args.jobs)]),
                             ('dram_adapter',[sys.executable,'scripts/check_dram_adapter.py'])]:
            verify(self.cohort_path)
            with(global_dir/(name+'.log')).open('w')as stream:subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        jobs=[(kind,stages,rows)for stages in self.args.stages for kind,rows in [('pe',1),('array',2),('array',8),('gemm',8)]]
        for kind,stages,rows in jobs:
            key=f'{kind}_{stages}_{rows}x{rows}'
            if key not in self.progress:self.update(key,status='pending')
        errors=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.workers)as executor:
            futures={executor.submit(self.configuration,*job):job for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                try:future.result()
                except Exception as error:errors.append(str(error))
        verify(self.cohort_path)
        summary=dict(passed=not errors and all(v['status']=='passed'for v in self.progress.values()),
                     configurations=self.progress,errors=errors,cohort_sha256=self.cohort['cohort_sha256'],
                     full_hardware_plan_complete=False,one_ghz_certified=False,
                     limitation='Functional/code campaign only. Storage/core/CDC physical signoff and 1GHz acceptance remain separate.')
        (self.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2))
        if errors:raise RuntimeError('campaign has failed configurations')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--stages',nargs='+',type=int,choices=[2],default=[2])
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--jobs',type=int,default=4)
    args=parser.parse_args();Campaign(args).run()
