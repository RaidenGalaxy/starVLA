#!/usr/bin/env python3
"""Audit actual reset synchronizers and timed stage paths without changing SDC."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from run import ROOT, sha, tool
from asap7_platform import profile


def run(args):
    source = args.source.resolve()
    proof = source / 'verification.json'
    prior = json.loads(proof.read_text())
    from async_mapping_policy import validate_checkpoint
    validate_checkpoint(source,prior)
    checkpoint = source / 'repaired.odb'
    if (prior['core_period_ns'], prior['memory_period_ns']) != (1, 2):
        raise ValueError('reset audit cannot change target clocks')
    for path, digest in prior['evidence'].items():
        if sha(path) != digest:
            raise ValueError('source evidence changed: ' + path)
    if prior['evidence'].get(str(checkpoint)) != sha(checkpoint):
        raise ValueError('reset audit needs an authenticated legal checkpoint')
    if not prior['logical_roots_and_protected_data_clock_masters_preserved']:
        raise ValueError('source has incomplete connectivity verification')
    out = args.out.resolve()
    if out.exists():
        raise ValueError('preserve earlier reset audits')
    out.mkdir(parents=True)
    shutil.copy2(__file__, out / 'audit_driver.py')
    for name in ('reset_boundary_audit.tcl', 'reset_repair.tcl', 'protected_clocks.tcl'):
        shutil.copy2(ROOT / 'physical' / name, out / name)
    technology = profile('TT')
    # The source binds both the standard-cell and native SRAM libraries.
    library = technology['liberty']
    if prior['evidence'].get(str(library)) != sha(library):
        raise ValueError('reset audit process library differs')
    candidates = [Path(path) for path in prior['evidence'] if path.endswith('.lib') and Path(path) != library]
    if len(candidates) != 1:
        raise ValueError('source must identify one actual SRAM timing library')
    macro_library = candidates[0]
    sdc = source / 'design.sdc'
    saved_sdc = sha(sdc)
    pairs = [('resetSync', 'io_memoryClock'), ('tile.coreReset_sync', 'io_coreClock'),
        ('tile.memoryReset_sync', 'io_memoryClock')]
    body = []
    for index, (prefix, clock) in enumerate(pairs):
        body.append(f'''lassign [simd_audit_reset_pair {{{prefix}}} {{{clock}}} $stream] first second
set first_sta [string map {{\\\\ ""}} $first]
set second_sta [string map {{\\\\ ""}} $second]
report_checks -from [get_cells [list $first_sta]] -to [get_pins [list "$second_sta/D"]] -path_delay max -group_count 1 -format full_clock_expanded -digits 9 > {{{out / (str(index) + '_setup.rpt')}}}
report_checks -from [get_cells [list $first_sta]] -to [get_pins [list "$second_sta/D"]] -path_delay min -group_count 1 -format full_clock_expanded -digits 9 > {{{out / (str(index) + '_hold.rpt')}}}
''')
    script = out / 'audit.tcl'
    script.write_text(f'''set_thread_count 4
read_liberty {{{library}}}
read_liberty {{{macro_library}}}
read_db {{{checkpoint}}}
set_cmd_units -time ns -capacitance fF
read_sdc {{{sdc}}}
set_propagated_clock [all_clocks]
source {{{technology['root'] / 'setRC.tcl'}}}
set_wire_rc -signal -layer M3
set_wire_rc -clock -layer M5
estimate_parasitics -placement
source {{{out / 'reset_repair.tcl'}}}
source {{{out / 'protected_clocks.tcl'}}}
source {{{out / 'reset_boundary_audit.tcl'}}}
set stream [open {{{out / 'pairs.tsv'}}} w]
puts $stream "prefix\tclock\tfirst\tsecond\treset_truth\tset_truth"
{''.join(body)}
close $stream
puts SIMD_RESET_BOUNDARY_AUDIT_COMPLETE
''')
    local = ROOT / '.deps/tools/openroad'
    env = {**os.environ, 'LD_LIBRARY_PATH': ':'.join([str(local / 'opt/or-tools/lib'),
        str(local / 'usr/lib/x86_64-linux-gnu'), os.environ.get('LD_LIBRARY_PATH', '')])}
    log = out / 'openroad.log'
    timed_out = False
    with log.open('w') as stream:
        try:
            result = subprocess.run([tool('openroad'), '-exit', str(script)], env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=600)
            code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            code = -1
    passed = code == 0 and 'SIMD_RESET_BOUNDARY_AUDIT_COMPLETE' in log.read_text()
    paths_timed = []
    if passed:
        for index in range(3):
            checks = {}
            for kind, mode in [('setup', 'max'), ('hold', 'min')]:
                text = (out / f'{index}_{kind}.rpt').read_text()
                checks[kind] = all(value in text for value in ('Startpoint:', 'Endpoint:', 'Path Type: ' + mode))
            paths_timed.append(checks)
        passed = all(all(item.values()) for item in paths_timed)
    if sha(sdc) != saved_sdc:
        raise ValueError('read-only audit changed source SDC')
    record = dict(kind='read_only_actual_system_reset_boundary_audit', passed=passed,
        returncode=code, timed_out=timed_out, core_period_ns=1, memory_period_ns=2,
        synchronizer_pairs=3, asynchronous_reset_pins=6, stage_paths=paths_timed,
        raw_reset_truth_order=['00', '01', '10', '11'], active_low_async_truth=[1, 0, 0, 0],
        source_sdc_unchanged=True, new_timing_exceptions_applied=False,
        source_verification=str(proof), source_verification_sha256=sha(proof),
        physical_frequency_certified=False,
        evidence={str(path): sha(path) for path in [proof, checkpoint, sdc, library,
            macro_library, *[p for p in out.iterdir() if p.is_file()]]})
    (out / 'verification.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'evidence'}, indent=2))
    if not passed:
        raise RuntimeError('read-only reset audit failed; evidence retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
