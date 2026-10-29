#!/usr/bin/env python3
"""Fetch immutable ASAP7 collateral and record its hashes; never certify it."""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORFS = '19dc41d9e83b8e6903d8408eb99c563a9470a1ad'
SRAM = '522eeccbccefcd66e61893fa1059df24d95e9f86'


def fetch(url, path, expected=None):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        temporary = path.with_suffix(path.suffix + '.part')
        temporary.write_bytes(content)
        temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and digest != expected:
        raise ValueError('ASAP7 collateral differs from lock: ' + str(path))
    return digest


def prepare(update_lock=False):
    lock_path = ROOT / 'dependencies.lock.json'
    lock = json.loads(lock_path.read_text())
    files = ['config.mk', 'setRC.tcl', 'lef/asap7_tech_1x_201209.lef',
             'lef/asap7sc7p5t_28_R_1x_220121a.lef',
             'gds/asap7sc7p5t_28_R_220121a.gds', 'rcx_patterns.rules',
             'openRoad/make_tracks.tcl', 'openRoad/tapcell.tcl',
             'openRoad/pdn/grid_strategy-M1-M2-M5-M6.tcl']
    for corner in ('TT', 'SS', 'FF'):
        for group, date in [('AO','211120'), ('INVBUF','220122'), ('OA','211120'),
                            ('SIMPLE','211120'), ('SEQ','220123')]:
            if group=='SIMPLE' and corner=='FF':date='250407'
            files.append(f'lib/NLDM/asap7sc7p5t_{group}_RVT_{corner}_nldm_{date}.lib' +
                         ('' if group == 'SEQ' else '.gz'))
    macro_files = ['README.md', 'LICENSE', 'verilog/README.txt', 'gds/srambank_64b.gds',
                   'generated/LEF/srambank_256x4x64_6t122.lef',
                   'generated/LIB/srambank_256x4x64_6t122.lib',
                   'generated/verilog/srambank_256x4x64_6t122.v']
    specs = [
        ('asap7', ORFS, 'https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/' + ORFS + '/flow/platforms/asap7/', files),
        ('asap7_sram', SRAM, 'https://raw.githubusercontent.com/The-OpenROAD-Project/asap7_sram_0p0/' + SRAM + '/', macro_files)]
    for name, commit, base, paths in specs:
        if not update_lock and name not in lock:
            raise ValueError('initial acquisition requires --update-lock')
        records = {}
        def one(relative):
            old = lock.get(name, {}).get('files', {}).get(relative, {})
            if not update_lock and 'sha256' not in old:
                raise ValueError('missing locked resource: ' + relative)
            digest = fetch(base + relative, ROOT / '.deps' / name / relative, old.get('sha256'))
            return relative, dict(path=relative, sha256=digest)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            for relative, record in pool.map(one, paths):
                records[relative] = record
        lock[name] = dict(commit=commit, base_url=base, files=records)
    if update_lock:
        lock_path.write_text(json.dumps(lock, indent=2) + '\n')
    # Keep only F2 arithmetic in active synthesis and simulation. Macro cells
    # are explicit instances and must not be mapped into standard cells.
    print('Verified ASAP7 standard-cell and candidate SRAM collateral')
    return lock


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--update-lock', action='store_true')
    prepare(parser.parse_args().update_lock)
