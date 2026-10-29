#!/usr/bin/env python3
"""Read-only audit of retained evidence; never promotes a simulation clock."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from runtime import validated_build
from simd_eval.area import load_area
from simd_eval.certification import core_certificate
from simd_eval.reporting import compare


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def build_audit(path):
    record = {'build': str(path.parent.relative_to(ROOT))}
    try:
        guard, caps = validated_build(path.parent)
        guard.close()
        record.update(authenticated=True, target=caps.get('target'), stages=caps['stages'],
                      rows=caps['rows'], cols=caps['cols'], rtl_sha256=caps['rtl_sha256'],
                      simulator_sha256=caps['simulator_sha256'],
                      inline_assertions=caps.get('inline_chisel_assertions', False),
                      missing_scope_metadata=['target'] if 'target' not in caps else [])
    except Exception as error:
        record.update(authenticated=False, error=str(error))
    return record


def area_audit(path):
    record = {'report': str(path.relative_to(ROOT)), 'report_sha256': sha(path)}
    try:
        proof = load_area(path)['proof']
        record.update(authenticated=True, scope=proof['scope'], stages=proof['stages'],
                      rows=proof['rows'], cols=proof['cols'],
                      cell_area_um2=proof['cell_area_um2'], rtl_sha256=proof['rtl_sha256'],
                      mapped_sha256=proof['mapped_sha256'],
                      full_storage=proof.get('memory_inventory', {}).get('all_nominal_bits_retained'))
    except Exception as error:
        record.update(authenticated=False, error=str(error))
    return record


def timing_audit(path):
    proof = json.loads(path.read_text())
    record = {'report': str(path.relative_to(ROOT)), 'report_sha256': sha(path)}
    record.update({key: proof.get(key) for key in (
        'scope', 'stages', 'rows', 'cols', 'period_ns', 'setup_wns_ns', 'hold_wns_ns',
        'tns_ns', 'electrical_clean', 'route_drc_clean', 'coverage_clean',
        'constraint_policy_clean', 'rtl_sha256', 'netlist_sha256', 'timing_status')})
    checks = {}
    for name, filename in proof.get('evidence', {}).items():
        source = Path(filename)
        expected = proof.get('evidence_sha256', {}).get(name)
        checks[name] = source.is_file() and expected is not None and sha(source) == expected
    record['declared_evidence_hashes'] = checks
    record['all_declared_evidence_hashes_match'] = bool(checks) and all(checks.values())
    numeric = path.parent / 'gate_verification_routed.json'
    if not numeric.exists():
        numeric = path.parent / 'gate_verification.json'
    record['numeric_kind'] = json.loads(numeric.read_text()).get('mapped_netlist_kind') if numeric.exists() else None
    if proof.get('timing_status') in ('kernel_closed', 'array_closed'):
        record['one_ghz_certificate'] = core_certificate(
            proof, json.loads(numeric.read_text()) if numeric.exists() else {})
    else:
        record['one_ghz_certificate'] = {'passed': False, 'failures': ['final timing not closed']}
    return record


def replay_pe(build, out):
    # Use a newly generated trace and the current independent rational oracle;
    # the executable and all RTL/harness inputs are authenticated separately.
    from check_pe import make_vectors, verify_output
    guard, caps = validated_build(build)
    commands, expected, acc, sticky = make_vectors()
    directory = out / ('replay_' + build.name)
    directory.mkdir(exist_ok=True)
    trace = directory / 'vectors.txt'
    trace.write_text(''.join(commands))
    result = subprocess.run([str(build / 'sim'), str(trace)], capture_output=True,
                            text=True, timeout=300)
    log = directory / 'simulation.log'
    log.write_text(result.stdout + result.stderr)
    record = {'build': str(build.relative_to(ROOT)), 'returncode': result.returncode,
              'rtl_sha256': caps['rtl_sha256'], 'simulator_sha256': caps['simulator_sha256'],
              'trace_sha256': sha(trace), 'log_sha256': sha(log)}
    try:
        if result.returncode:
            raise RuntimeError('simulator failed; see frozen replay log')
        count = verify_output(result.stdout, expected, acc, sticky,
                              caps['stages'] + caps.get('pe_state_read_latency', 0) + caps.get('pe_commit_latency', 0))
        record.update(passed=True, vectors=count, all_formats=True)
    except Exception as error:
        record.update(passed=False, error=str(error))
    guard.close()
    return record


def metric_audit(path, authenticated_areas):
    data = json.loads(path.read_text())
    seconds = data.get('elapsed_ps', 0) * 1e-12
    record = {'report': str(path.relative_to(ROOT)), 'report_sha256': sha(path),
              'timing_validated': data.get('timing_validated'), 'checks': {}}
    if seconds <= 0:
        record.update(passed=False, error='missing or nonpositive elapsed time')
        return record
    formulas = [('gmac_s', 'macs'), ('useful_bandwidth_GB_s', 'useful_transfer_bytes'),
                ('dram_bandwidth_GB_s', 'dram_physical_bytes'),
                ('dma_bandwidth_GB_s', 'dma_transfer_bytes')]
    for field, count in formulas:
        if data.get(field) is not None and count in data:
            record['checks'][field] = math.isclose(data[field], data[count] / seconds / 1e9,
                                                   rel_tol=1e-10, abs_tol=1e-12)
    if data.get('gops') is not None and 'macs' in data:
        record['checks']['gops'] = math.isclose(data['gops'], 2 * data['macs'] / seconds / 1e9,
                                               rel_tol=1e-10, abs_tol=1e-12)
    if data.get('sequencer_cycle_accounting_exact') and 'sequencer_cycle_breakdown' in data:
        record['checks']['cycle_category_sum'] = (sum(data['sequencer_cycle_breakdown'].values())
                                                  == data['sequencer_memory_cycles'])
    area = data.get('area_netlist_sha256')
    if area:
        record['checks']['bound_area_netlist_authenticates'] = area in authenticated_areas
    record['assessable'] = bool(record['checks'])
    record['status'] = ('passed' if all(record['checks'].values()) else 'failed') if record['checks'] else 'missing_derived_fields'
    record['passed'] = bool(record['checks']) and all(record['checks'].values())
    return record


def replay_fp64(out):
    from simd_eval.reference import fp_operation, Format, Op
    directory = ROOT / '.build/physical/fp64_12_1x1_1ns_split_alignment_hold80'
    trace = out / 'fp64_routed_replay_vectors.txt'
    # Recompute every stored expected result using the current rational oracle.
    commands = []
    for line in (directory / 'gate_vectors.txt').read_text().splitlines():
        a, b, c, op, old_result, old_flags = line.split()
        value, flags = fp_operation(int(a, 16), int(b, 16), int(c, 16), Format.F64, Op(int(op)))
        if (value, flags) != (int(old_result, 16), int(old_flags)):
            raise AssertionError('stored FP64 expected value disagrees with current oracle')
        commands.append(f'{a} {b} {c} {op} {value:x} {flags}\n')
    trace.write_text(''.join(commands))
    binary = directory / 'gate_sim'
    result = subprocess.run([str(binary), str(trace), '12', 'free'],
                            capture_output=True, text=True, timeout=120)
    log = out / 'fp64_routed_replay.log'
    log.write_text(result.stdout + result.stderr)
    evidence = json.loads(result.stdout) if result.returncode == 0 else {}
    return {'passed': result.returncode == 0 and evidence.get('passed') is True
                     and evidence.get('gate_vectors') == 1200,
            'vectors': len(commands), 'executable_sha256': sha(binary),
            'routed_netlist_sha256': sha(directory / 'routed.v'), 'trace_sha256': sha(trace),
            'log_sha256': sha(log),
            'compile_dependency_manifest_sha256': sha(directory / 'gate_obj/VGateDut__verFiles.dat'),
            'recompiled': False, 'scope': 'existing_fp64_routed_gate_executable_replay'}


def replay_alu(out):
    from check_alu import make_vectors
    build = ROOT / '.build/alu_2_1x1'
    guard, caps = validated_build(build)
    commands, expected = make_vectors(count=200, exhaustive=True)
    trace = out / 'f2_alu_exhaustive_vectors.txt'
    trace.write_text(''.join(commands))
    records = []
    for stalled in (False, True):
        args = [str(build / 'sim'), str(trace)] + (['101'] if stalled else [])
        result = subprocess.run(args, capture_output=True, text=True, timeout=300)
        log = out / ('f2_alu_stalled.log' if stalled else 'f2_alu_unstalled.log')
        log.write_text(result.stdout + result.stderr)
        record = {'random_backpressure': stalled, 'log_sha256': sha(log)}
        try:
            if result.returncode:
                raise RuntimeError('ALU executable failed')
            seen = set()
            for line in result.stdout.splitlines():
                if not line or not line[0].isdigit():
                    continue
                tag, data, flags, overflow, illegal, start, end = line.split()
                index = int(tag)
                if index in seen or index >= len(expected):
                    raise AssertionError('duplicate/unexpected ALU tag')
                seen.add(index)
                if (int(data, 16), int(flags, 16), int(overflow, 16)) != expected[index] or int(illegal):
                    raise AssertionError('ALU result or flags disagree with independent reference')
                if not stalled and int(end) - int(start) != 2:
                    raise AssertionError('F2 arithmetic latency differs from two clocks')
            if len(seen) != len(expected):
                raise AssertionError('missing ALU outputs')
            record.update(passed=True, vectors=len(seen))
        except Exception as error:
            record.update(passed=False, error=str(error))
        records.append(record)
    guard.close()
    return {'passed': all(x['passed'] for x in records), 'rtl_sha256': caps['rtl_sha256'],
            'simulator_sha256': caps['simulator_sha256'], 'trace_sha256': sha(trace),
            'exhaustive_int4_fp4_fp8_pairs': True, 'runs': records}


def audit(out):
    out.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc).isoformat()
    summary = json.loads((ROOT / 'results/summary.json').read_text())
    inventory = []
    for entry in summary['reports']:
        path = ROOT / entry['source']
        raw = path.read_bytes()
        current = json.loads(raw)
        inventory.append({'source': entry['source'], 'sha256': hashlib.sha256(raw).hexdigest(),
                          'summary_snapshot_still_matches': hashlib.sha256(raw).hexdigest() == entry['sha256'],
                          'scope': current.get('scope'), 'passed_field': current.get('passed'),
                          'status': current.get('timing_status', current.get('status')),
                          'rtl_sha256': current.get('rtl_sha256'),
                          'timing_validated': current.get('timing_validated')})
    caps = sorted((ROOT / '.build').glob('*/capabilities.json'))
    synth = sorted((ROOT / '.build/physical').glob('*/synthesis.json'))
    timing = sorted((ROOT / '.build/physical').glob('*/timing.json'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        builds = list(pool.map(build_audit, caps))
        areas = list(pool.map(area_audit, synth))
        clocks = list(pool.map(timing_audit, timing))
    print('AUTHENTICATED', sum(x['authenticated'] for x in builds), '/', len(builds),
          'builds;', sum(x['authenticated'] for x in areas), '/', len(areas), 'areas', flush=True)
    comparisons = []
    for folder in ['actual_gate_crop_f2_8x8_full_storage', 'actual_full_decode_full_storage_f2',
                   'actual_all_decode_f2', 'actual_all_prefill_f2']:
        directory = ROOT / 'artifacts' / folder
        baseline = json.loads((directory / 'BF16/metrics.json').read_text())
        for metric in sorted(directory.glob('*/metrics.json')):
            if metric.parent.name == 'BF16':
                continue
            candidate = json.loads(metric.read_text())
            item = {'baseline': str((directory / 'BF16/metrics.json').relative_to(ROOT)),
                    'candidate': str(metric.relative_to(ROOT))}
            try:
                recomputed = compare(baseline, candidate)
                saved = directory / (metric.parent.name + '_comparison.json')
                stored = json.loads(saved.read_text())
                if abs(stored['speedup'] - recomputed['speedup']) > 1e-12:
                    raise ValueError('stored speedup differs from recomputed elapsed-time ratio')
                item.update(passed=True, speedup=recomputed['speedup'],
                            time_kind=recomputed['time_kind'], rtl_sha256=baseline['rtl_sha256'],
                            rows=baseline['rows'], cols=baseline['cols'],
                            mnk=[baseline.get(k) for k in ('m', 'n', 'k')],
                            has_bound_area='area_netlist_sha256' in baseline)
            except Exception as error:
                item.update(passed=False, error=str(error))
            comparisons.append(item)
    authenticated_areas = {x.get('mapped_sha256') for x in areas if x['authenticated']}
    metrics = [metric_audit(path, authenticated_areas)
               for path in sorted((ROOT / 'artifacts').rglob('metrics.json'))]
    replays = []
    for name in ['pe_2_1x1', 'pe_8_1x1_distributed_capture25', 'pe_12_1x1_distributed_capture25']:
        record = replay_pe(ROOT / '.build' / name, out)
        replays.append(record)
        print('REPLAY', name, record.get('passed'), record.get('error', ''), flush=True)
    stale_reports = []
    for path in sorted((ROOT / '.build').glob('*/verification*.json')):
        data = json.loads(path.read_text())
        capfile = path.parent / 'capabilities.json'
        if capfile.exists() and data.get('rtl_sha256'):
            cap = json.loads(capfile.read_text())
            if data['rtl_sha256'] != cap.get('rtl_sha256'):
                stale_reports.append({'report': str(path.relative_to(ROOT)),
                                      'reported_rtl': data['rtl_sha256'],
                                      'directory_rtl': cap.get('rtl_sha256')})
    result = {'audit_started_utc': start, 'audit_completed_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'retained_evidence_audit', 'report_inventory': inventory,
              'build_authentication': builds, 'synthesis_area_authentication': areas,
              'final_timing_authentication': clocks, 'comparison_recomputation': comparisons,
              'independent_numeric_replays': replays,
              'fp64_routed_executable_replay': replay_fp64(out),
              'metric_formula_audits': metrics,
              'stale_direct_build_reports': stale_reports,
              'f2_alu_exhaustive_replay': replay_alu(out),
              'counts': {'inventoried_reports': len(inventory), 'builds': len(builds),
                         'authenticated_builds': sum(x['authenticated'] for x in builds),
                         'area_reports': len(areas), 'authenticated_area_reports': sum(x['authenticated'] for x in areas),
                         'final_timing_reports': len(clocks),
                         'valid_one_ghz_certificates': sum(x['one_ghz_certificate']['passed'] for x in clocks),
                         'passed_comparison_recomputations': sum(x['passed'] for x in comparisons),
                         'comparison_recomputations': len(comparisons)},
              'limitations': ['This authenticates retained artifacts and selected independent replays; it does not rerun every historical workload.',
                              'A passed report field alone is not a plan acceptance item.',
                              'Frozen historical builds do not certify the latest source or a different physical netlist.']}
    result['counts'].update(metric_reports=len(metrics),
                            metric_formula_audits_assessable=sum(x.get('assessable', False) for x in metrics),
                            metric_formula_audits_passed=sum(x['passed'] for x in metrics),
                            stale_direct_build_reports=len(stale_reports))
    result['audit_completed_utc'] = datetime.now(timezone.utc).isoformat()
    result['audit_script_sha256'] = sha(Path(__file__))
    (out / 'evidence_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['counts'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/plan_audit_20261025')
    audit(parser.parse_args().out.resolve())
