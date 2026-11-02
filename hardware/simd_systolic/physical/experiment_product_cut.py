#!/usr/bin/env python3
"""Verify isolated F2 product candidates and resume authenticated import failures."""
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


def run(args):
    label=args.label;out=args.out.resolve()
    work=ROOT/('.build/experiments/'+label)
    raw_label=label+('_raw_recovered'if args.resume_source else'_raw')
    raw=ROOT/('.build/physical/asap7_TT_int64_2_1x1_1ns_'+raw_label)
    clean=ROOT/('.build/physical/asap7_TT_int64_2_1x1_1ns_'+label)
    if any(p.exists()for p in(out,raw,clean))or(not args.resume_source and work.exists()):
        raise ValueError('preserve previous product-cut results; choose new paths')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'sequence_driver.py')
    state=dict(kind='isolated_four_term_product_cut'if args.product_terms==4 else'isolated_two_term_product',
        arithmetic_stages=2,core_period_ns=1,product_kind=args.product_kind,
        first_boundary_product_terms=args.product_terms,local_enable=True,main_source_changed=False,
        physical_frequency_certified=False,steps=[],passed=False)
    def save():
        p=out/'state.json.pending';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(out/'state.json')
    def phase(name,command,timeout):
        state['active_phase']=name;save();timed_out=False;log=out/(name+'.log')
        with log.open('w')as stream:
            process=subprocess.Popen([str(x)for x in command],cwd=ROOT,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
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
            timed_out=timed_out,timeout_seconds=timeout,log=str(log),log_sha256=sha(log)))
        save()
        if code or timed_out:raise RuntimeError('bounded phase failed: '+name)
    py=sys.executable
    try:
        if args.resume_source:
            source=args.resume_source.resolve();parent_path=source/'verification.json'
            parent=json.loads(parent_path.read_text());functional=json.loads((source/'functional/verification.json').read_text())
            if(parent['arithmetic_stages'],parent['core_period_ns'],parent['first_boundary_product_terms'])!=(2,1,args.product_terms):
                raise ValueError('resume source changes stages,period or product boundary')
            if parent.get('product_kind','booth')!=args.product_kind or not functional['functional_passed']:
                raise ValueError('resume source has another product or no complete functional verification')
            if Path(functional['work_directory']).resolve()!=work:
                raise ValueError('resume label differs from the original qualified source work')
            for step in parent['steps']:
                if sha(step['log'])!=step['log_sha256']:raise ValueError('resume execution log changed')
            steps={s['name']:s for s in parent['steps']}
            if any(steps[n]['returncode']or steps[n]['timed_out']for n in('functional','mapping')):
                raise ValueError('resume source did not finish functional verification and mapping')
            mapping=steps['mapping']['command'];source_label=mapping[mapping.index('--label')+1]
            mapped=ROOT/('.build/physical/asap7_TT_int64_2_1x1_1ns_'+source_label)
            identity=json.loads((mapped/'build_identity.json').read_text())
            if(identity['scope'],identity['stages'],identity['analysis_period_ns'])!=('int64',2,1):
                raise ValueError('resume mapping has another scope or period')
            if sha(mapped/'synth.v')!=identity['mapped_sha256']or sha(identity['liberty_file'])!=identity['liberty_sha256']:
                raise ValueError('resume mapping or library changed')
            failed=steps.get('raw_mapped_numerical',{})
            error=mapped/'check_mapped/build.log'
            if not failed.get('returncode')or failed.get('timed_out')or'Too many preprocessor tokens on a line'not in error.read_text():
                raise ValueError('only the preserved netlist lexer failure can be resumed here')
            shutil.copytree(mapped,raw,ignore=shutil.ignore_patterns('check_*','.lock'))
            shutil.copytree(source/'functional',out/'functional')
            state.update(functional_passed=True,resumed_source=str(source),
                resumed_source_sha256=sha(parent_path),mapping_identity_sha256=sha(mapped/'build_identity.json'),
                original_mapping_sha256=sha(mapped/'synth.v'),original_compile_failure_sha256=sha(error))
            save()
        else:
            phase('functional',[py,'physical/experiment_pe_context.py','--work',work,
                '--out',out/'functional','--fpu-snapshot',args.fpu_snapshot.resolve(),
                '--integer-conversion-onehot','--registered-integer-prefixes',
                '--balanced-integer-select','--late-integer-mask','--integer-product-pair',
                '--integer-product-kind',args.product_kind,'--integer-product-terms',str(args.product_terms),
                '--local-enable','--physical-target','int64','--functional-only','-j','4'],7200)
            state['functional_passed']=True;save()
            phase('mapping',[py,'physical/run.py','--target','int64','--stages','2','--rows','1',
                '--period','1','--label',raw_label,'--rtl-build',work/'.build/alu_2_1x1_ctx_word',
                '--mapping-profile','fast','--synth-only'],2400)
        phase('raw_mapped_numerical',[py,'physical/check_int64.py',raw,'--mapped','--count','1000'],1800)
        phase('observable_cleanup',[py,'physical/cleanup_mapped_component.py',
            '--source',raw,'--out',clean],2400)
        phase('cts',[py,'physical/run.py','--target','int64','--stages','2','--rows','1',
            '--period','1','--label',label,'--reuse-mapped','--clock-profile','standard',
            '--placement-profile','geometric','--scalarize-tool-input','--cts-only'],1800)
        # Keep the pre-scalarization numerical result and verify the actual
        # OpenSTA import separately, even though splitnets only changes wires.
        archive=clean/'check_mapped_pre_scalarization'
        (clean/'check_mapped').rename(archive)
        state['pre_scalarization_numerical']=dict(path=str(archive/'verification.json'),
            sha256=sha(archive/'verification.json'));save()
        phase('actual_import_numerical',[py,'physical/check_int64.py',clean,'--mapped','--count','1000'],1800)
        timing=json.loads((clean/'cts_timing.json').read_text())
        proof=json.loads((clean/'check_mapped/verification.json').read_text())
        if not proof['passed']or proof['netlist_sha256']!=timing['mapped_sha256']or proof['simulated_netlist_sha256']!=sha(clean/'tool_input.v'):
            raise ValueError('CTS and actual numerical import identity differ')
        state.update(passed=True,setup_wns_ns=timing['setup_wns_ns'],hold_wns_ns=timing['hold_wns_ns'],
            timing=str(clean/'cts_timing.json'),timing_sha256=sha(clean/'cts_timing.json'),
            actual_import_numerical_sha256=sha(clean/'check_mapped/verification.json'))
    except Exception as error:
        state['error']=str(error);raise
    finally:
        state.pop('active_phase',None);save()
        (out/'verification.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--label',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--fpu-snapshot',type=Path)
    p.add_argument('--product-kind',choices=['booth','baugh'],default='booth')
    p.add_argument('--product-terms',type=int,choices=[2,4],default=4)
    p.add_argument('--resume-source',type=Path,help='reuse completed functional/mapping evidence after a preserved netlist lexer failure')
    a=p.parse_args()
    if not re.fullmatch('[a-z0-9_]+',a.label):p.error('use lowercase letters,digits,underscore')
    if a.product_terms==4 and a.product_kind!='booth':p.error('four-term candidate currently uses Booth')
    if not a.resume_source and not a.fpu_snapshot:p.error('a fresh candidate requires --fpu-snapshot')
    run(a)
