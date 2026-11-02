#!/usr/bin/env python3
"""Recheck existing routed geometry/SPEF with a proven zero-fanout census."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from run import ROOT,sha
from parasitics import audit


def run(args):
    source=args.source.resolve();out=args.out.resolve();driver_audit=args.driver_audit.resolve()
    identity=json.loads((source/'build_identity.json').read_text())
    if (identity['scope'],identity['stages'],identity['analysis_period_ns'])!=('int64',2,1):
        raise ValueError('requires F2 I64 at the original1ns target')
    if sha(identity['liberty_file'])!=identity['liberty_sha256']:raise ValueError('process library changed')
    coverage=audit((source/'route.log').read_text(),source/'design.spef',driver_audit)
    if not coverage['clean']:raise ValueError('connected-driver annotation is incomplete')
    numerical_path=source/'check_routed/verification.json';numerical=json.loads(numerical_path.read_text())
    if not numerical['passed']or numerical['rtl_sha256']!=identity['rtl_sha256']or numerical['netlist_sha256']!=sha(source/'routed.v'):
        raise ValueError('requires actual numerical proof for this same routed netlist')
    if out.exists():raise ValueError('preserve previous rechecks')
    shutil.copytree(source,out,ignore=shutil.ignore_patterns('check_*','*.def','recovery_state.json',
        'recovery_verification.json','numerical_execution.json','.lock','timing.json'))
    shutil.copy2(__file__,out/'recheck_driver.py')
    shutil.copy2(driver_audit,out/'parasitic_driver_audit.json')
    state=dict(kind='same_geometry_routed_sta_recheck',passed=False,physical_frequency_certified=False,
        source=str(source),source_identity_sha256=sha(source/'build_identity.json'),
        source_numerical_verification=str(numerical_path),source_numerical_sha256=sha(numerical_path),
        stages=2,core_period_ns=1,reused_routing_and_spef=True,routed_numerical_passed=True)
    try:
        log=out/'recheck.log'
        with log.open('w')as stream:
            p=subprocess.Popen([sys.executable,str(ROOT/'physical/finalize.py'),str(out)],cwd=ROOT,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);raise RuntimeError('STA recheck timed out')
        if code:raise RuntimeError('STA recheck failed; evidence retained')
        timing=json.loads((out/'timing.json').read_text())
        if not timing['parasitic_annotation_clean']:raise ValueError('annotation failed on fresh STA read')
        if sha(out/'routed.v')!=numerical['netlist_sha256']or sha(out/'design.spef')!=sha(source/'design.spef'):
            raise ValueError('STA/numerical implementation changed')
        state.update(passed=True,timing_sha256=sha(out/'timing.json'),
            setup_wns_ns=timing['setup_wns_ns'],hold_wns_ns=timing['hold_wns_ns'],tns_ns=timing['tns_ns'],
            setup_hold_closed=timing['setup_hold_closed'],parasitic_annotation_clean=True)
    except Exception as error:
        state['error']=str(error);raise
    finally:
        state['evidence']={str(p):sha(p)for p in [numerical_path,driver_audit,out/'recheck_driver.py',
            *[out/n for n in('route.log','recheck.log','design.spef','routed.v','timing.json')if(out/n).exists()]]}
        (out/'recheck_verification.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--driver-audit',type=Path,required=True);run(parser.parse_args())
