#!/usr/bin/env python3
"""Run bounded I64 routing, extraction and numerical checks as separate phases."""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from run import ROOT, sha


def run(args):
    out = args.out.resolve()
    if out.exists():
        raise ValueError('preserve previous route sequences; choose a new output')
    out.mkdir(parents=True)
    shutil.copy2(__file__, out / 'sequence_driver.py')
    state = dict(kind='bounded_separate_i64_route_extraction', stages=2,
        core_period_ns=1, physical_frequency_certified=False, steps=[], passed=False)
    def save():
        pending = out / 'state.json.pending'
        pending.write_text(json.dumps(state, indent=2) + '\n')
        pending.replace(out / 'state.json')

    def phase(name, command, timeout=None, allow_unclassified_drivers=False):
        log = out / (name + '.log')
        state['active_phase'] = name
        save()
        timed_out = False
        with log.open('w') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
            state['active_pid'] = process.pid
            save()
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait()
        state['steps'].append(dict(name=name, command=command, returncode=code,
            timed_out=timed_out, log=str(log), log_sha256=sha(log)))
        state.pop('active_pid', None)
        save()
        if code and not timed_out and allow_unclassified_drivers:
            record=json.loads((final/'recovery_verification.json').read_text())
            if record.get('error')=='routed STA has no complete, error-free parasitic annotation':
                state['raw_export_requires_driver_audit']=True;save();return
        if code or timed_out:
            raise RuntimeError('bounded phase failed: ' + name)

    try:
        if args.wait_for_mapped_seconds:
            state['active_phase']='waiting_for_actual_mapped_proof';save()
            deadline=time.monotonic()+args.wait_for_mapped_seconds
            while True:
                try:
                    if args.producer_verification and args.producer_verification.exists():
                        producer=json.loads(args.producer_verification.read_text())
                        if not producer['passed']:raise RuntimeError('mapped producer failed; source evidence retained')
                    source=args.source.resolve()
                    checked=json.loads((source/'check_mapped/verification.json').read_text())
                    identity=json.loads((source/'build_identity.json').read_text())
                    if checked['passed']and checked['netlist_sha256']==sha(source/'synth.v')and checked['simulated_netlist_sha256']==sha(source/'tool_input.v')and(source/'cts_timing.json').exists():break
                except(FileNotFoundError,json.JSONDecodeError):pass
                if time.monotonic()>=deadline:raise RuntimeError('actual mapped proof did not arrive within bound')
                time.sleep(5)
        first_label = args.label + '_geometry'
        final_label = args.label + ('_export'if args.canonical_export else'_extracted')
        first = ROOT / ('.build/physical/asap7_TT_int64_2_1x1_1ns_' + first_label)
        final = ROOT / ('.build/physical/asap7_TT_int64_2_1x1_1ns_' + final_label)
        state.update(geometry_directory=str(first), extracted_directory=str(final))
        # Each route driver owns and bounds its own private tool process group.
        phase('detailed_route', [sys.executable, str(ROOT / 'physical/route_arithmetic.py'),
            '--source', str(args.source.resolve()), '--label', first_label,
            *([]if args.from_cts else['--resume-global-route']),
            '--detailed-route-only', '--timeout', str(args.route_timeout)])
        checkpoint = first / 'detailed_route_checkpoint.json'
        state['geometry_checkpoint_sha256'] = sha(checkpoint)
        save()
        phase('resume_guard', [sys.executable, str(ROOT / 'physical/check_arithmetic_resume.py'),
            '--source', str(first), '--out', str(out / 'resume_guard'), '--detailed-route'], timeout=300)
        if args.canonical_export:
            phase('canonical_export_and_extraction', [sys.executable,str(ROOT/'physical/recover_routed_export.py'),
                '--source',str(first),'--out',str(final),'--timeout',str(args.extraction_timeout)],
                allow_unclassified_drivers=True)
        else:
            phase('extracted_sta', [sys.executable, str(ROOT / 'physical/route_arithmetic.py'),
                '--source', str(first), '--label', final_label,
                '--resume-detailed-route', '--timeout', str(args.extraction_timeout)])
        timing = json.loads((final / 'timing.json').read_text())
        state.update(timing_sha256=sha(final / 'timing.json'),
            setup_wns_ns=timing['setup_wns_ns'], hold_wns_ns=timing['hold_wns_ns'],
            extracted_sta_completed=timing.get('parasitic_annotation_clean',False))
        if not(final/'check_routed/verification.json').exists():
            phase('routed_numerical', [sys.executable, str(ROOT / 'physical/check_int64.py'),
                str(final), '--routed', '--count', '1000'], timeout=1800)
        proof = final / 'check_routed/verification.json'
        checked = json.loads(proof.read_text())
        if (not checked['passed'] or checked['rtl_sha256'] != timing['rtl_sha256']
                or checked['netlist_sha256'] != sha(final / 'routed.v')):
            raise ValueError('routed numerical evidence differs from extracted implementation')
        if args.canonical_export:
            audit=out/'driver_audit';qualified=ROOT/('.build/physical/asap7_TT_int64_2_1x1_1ns_'+args.label+'_qualified')
            phase('parasitic_driver_audit',[sys.executable,str(ROOT/'physical/audit_parasitic_drivers.py'),
                '--directory',str(final),'--out',str(audit)],timeout=1800)
            phase('qualified_sta',[sys.executable,str(ROOT/'physical/recheck_routed_sta.py'),
                '--source',str(final),'--driver-audit',str(audit/'verification.json'),
                '--out',str(qualified)],timeout=2400)
            timing=json.loads((qualified/'timing.json').read_text())
            state.update(qualified_directory=str(qualified),extracted_sta_completed=True,
                parasitic_annotation_clean=timing['parasitic_annotation_clean'],
                setup_wns_ns=timing['setup_wns_ns'],hold_wns_ns=timing['hold_wns_ns'],
                tns_ns=timing['tns_ns'],setup_hold_closed=timing['setup_hold_closed'],
                qualified_verification_sha256=sha(qualified/'recheck_verification.json'))
        if not timing.get('parasitic_annotation_clean',False):
            raise ValueError('final STA still has incomplete or erroneous parasitic annotation')
        state.update(passed=True, routed_numerical_sha256=sha(proof))
    except Exception as error:
        state['error'] = str(error)
        raise
    finally:
        state.pop('active_phase', None)
        save()
        (out / 'verification.json').write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--from-cts',action='store_true',help='start from an authenticated CTS checkpoint')
    parser.add_argument('--canonical-export',action='store_true',help='prove safe internal naming and classify every unannotated driver before fresh STA')
    parser.add_argument('--wait-for-mapped-seconds',type=int,default=0)
    parser.add_argument('--producer-verification',type=Path)
    parser.add_argument('--route-timeout', type=int, default=7200)
    parser.add_argument('--extraction-timeout', type=int, default=7200)
    args = parser.parse_args()
    if not re.fullmatch('[a-z0-9_]+', args.label):
        parser.error('label must use lowercase letters, digits or underscores')
    if any(not 1 <= value <= 7200 for value in (args.route_timeout, args.extraction_timeout)):
        parser.error('each physical phase must have a timeout within1..7200seconds')
    if not 0<=args.wait_for_mapped_seconds<=3600:parser.error('mapped wait must be within0..3600seconds')
    run(args)
