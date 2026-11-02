#!/usr/bin/env python3
"""Correct all legacy asynchronous pins in a copied, authenticated placed system."""
import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from run import ROOT,sha,tool
from asap7_platform import profile
from async_mapping_policy import count_async,qualify


def netlist_records(path,swap=False):
    with Path(path).open()as stream:
        for line in stream:
            if re.match(r'\s*DFFASRHQNx1_ASAP7_75t_R\s',line):
                block=line
                while ');'not in block:
                    block+=next(stream)
                if swap:
                    reset=re.findall(r'\.RESETN\(([^)]*)\)',block)
                    preset=re.findall(r'\.SETN\(([^)]*)\)',block)
                    if len(reset)!=1 or len(preset)!=1:raise ValueError('malformed actual async instance')
                    block=re.sub(r'\.RESETN\([^)]*\)',lambda _:'.RESETN('+preset[0]+')',block)
                    block=re.sub(r'\.SETN\([^)]*\)',lambda _:'.SETN('+reset[0]+')',block)
                yield ''.join(block.split())
            else:
                normalized=''.join(line.split())
                if normalized:yield normalized


def run(args):
    source=args.source.resolve();proof=source/'verification.json'
    prior=json.loads(proof.read_text());checkpoint=source/'repaired.odb'
    mapping=args.legacy_mapping.resolve();mapped_identity=json.loads((mapping/'build_identity.json').read_text())
    if mapped_identity.get('async_mapping_policy')or mapped_identity['mapped_sha256']!=prior['source_mapped_sha256']:
        raise ValueError('not the authenticated legacy mapping')
    if sha(mapping/'synth.v')!=mapped_identity['mapped_sha256']:raise ValueError('legacy mapped netlist changed')
    mapping_log=(mapping/'synth.log').read_text()
    if not re.search(r'DFFASRHQNx1_ASAP7_75t_R.*_DFFSR_PNN_.*\.QN\(~Q\).*\.RESETN\(\s*S\).*\.SETN\(\s*R\)',mapping_log):
        raise ValueError('source does not exhibit the reproduced legacy async mapping defect')
    if prior.get('async_mapping_correction_verified')or prior.get('qualified_async_mapping_root'):
        raise ValueError('already qualified; never swap correct async pins')
    if not prior.get('passed')or not prior.get('logical_roots_and_protected_data_clock_masters_preserved'):
        raise ValueError('requires a completed legal source checkpoint')
    if(prior['core_period_ns'],prior['memory_period_ns'])!=(1,2):raise ValueError('target clocks differ')
    for name,digest in prior['evidence'].items():
        if sha(name)!=digest:raise ValueError('source evidence changed: '+name)
    if prior['evidence'].get(str(checkpoint))!=sha(checkpoint):raise ValueError('source checkpoint is not authenticated')
    count=count_async(source/'repaired.v')
    out=args.out.resolve()
    if out.exists():raise ValueError('preserve previous async-correction experiments')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'correction_driver.py')
    technology=profile('TT');library=technology['liberty']
    if prior['evidence'].get(str(library))!=sha(library):raise ValueError('source process library differs')
    macros=[Path(p)for p in prior['evidence']if p.endswith('.lib')and Path(p)!=library]
    if len(macros)!=1:raise ValueError('source must bind one SRAM library')
    async_identity=qualify(out,library)
    for name in('reset_repair.tcl','reset_boundary_audit.tcl','protected_clocks.tcl','correct_async_pins.tcl'):
        shutil.copy2(ROOT/'physical'/name,out/name)
    for name in('cdc_constraints.tcl','protected_power.tcl'):shutil.copy2(source/name,out/name)
    # SDC differs only in its absolute helper-file location.
    (out/'design.sdc').write_text((source/'design.sdc').read_text().replace(str(source),str(out)))
    script=out/'correct.tcl';script.write_text(f'''set_thread_count 8
read_liberty {{{library}}}
read_liberty {{{macros[0]}}}
read_db {{{checkpoint}}}
source {{{out/'reset_repair.tcl'}}}
source {{{out/'correct_async_pins.tcl'}}}
write_verilog {{{out/'before.v'}}}
simd_async_geometry {{{out/'before_geometry.tsv'}}}
simd_correct_async_pins {count} {{{out/'pin_swaps.tsv'}}}
simd_async_geometry {{{out/'after_geometry.tsv'}}}
write_db {{{out/'repaired.odb'}}}
write_verilog {{{out/'repaired.v'}}}
source {{{out/'protected_clocks.tcl'}}}
source {{{out/'reset_boundary_audit.tcl'}}}
set stream [open {{{out/'reset_pairs.tsv'}}} w]
simd_audit_reset_pair resetSync io_memoryClock $stream
simd_audit_reset_pair tile.coreReset_sync io_coreClock $stream
simd_audit_reset_pair tile.memoryReset_sync io_memoryClock $stream
close $stream
puts SIMD_ASYNC_CORRECTION_COMPLETE
''')
    local=ROOT/'.deps/tools/openroad'
    env={**os.environ,'LD_LIBRARY_PATH':':'.join([str(local/'opt/or-tools/lib'),
        str(local/'usr/lib/x86_64-linux-gnu'),os.environ.get('LD_LIBRARY_PATH','')])}
    with(out/'openroad.log').open('w')as stream:
        result=subprocess.run([tool('openroad'),'-exit',str(script)],stdout=stream,
            stderr=subprocess.STDOUT,env=env,timeout=args.timeout)
    if result.returncode or 'SIMD_ASYNC_CORRECTION_COMPLETE'not in(out/'openroad.log').read_text():
        raise RuntimeError('async correction failed; source and failure evidence retained')
    if sha(out/'before_geometry.tsv')!=sha(out/'after_geometry.tsv'):
        raise ValueError('async correction changed cell placement or masters')
    compared=0
    for before,after in itertools.zip_longest(netlist_records(out/'before.v',True),netlist_records(out/'repaired.v')):
        if before!=after:raise ValueError('correction changed non-async connectivity at record '+str(compared))
        compared+=1
    if count_async(out/'repaired.v')!=count:raise ValueError('async inventory changed')
    correction=dict(kind='all_legacy_async_pins_corrected_in_copied_system',passed=True,
        liberty_sha256=sha(library),legacy_mapping=str(mapping),
        asynchronous_cells_corrected=count,source_verification=str(proof),source_verification_sha256=sha(proof),
        all_non_async_connectivity_and_placement_preserved=True,netlist_records_compared=compared,
        only_change='swap RESETN/SETN on every verified single-reset DFFASRHQN cell',
        actual_liberty_event_proof=async_identity,reset_synchronizer_pairs_checked=3,
        source_sdc_semantics_preserved=True,physical_frequency_certified=False,
        full_system_gate_functional_verified=False,
        evidence={str(p):sha(p)for p in [proof,checkpoint,library,macros[0],mapping/'build_identity.json',mapping/'synth.v',mapping/'synth.log',
            *[p for p in out.iterdir()if p.is_file()]]})
    correction_file=out/'async_correction.json'
    correction_file.write_text(json.dumps(correction,indent=2)+'\n')
    record=dict(kind='corrected_async_placed_system_checkpoint',passed=True,core_period_ns=1,memory_period_ns=2,
        async_mapping_correction_verified=True,async_correction_verification=str(correction_file),
        async_correction_verification_sha256=sha(correction_file),
        source_verification=str(proof),source_verification_sha256=sha(proof),
        source_rtl_sha256=prior['source_rtl_sha256'],source_mapped_sha256=prior['source_mapped_sha256'],
        source_mapped_identity_is_legacy=True,physical_macro_package_manifest_sha256=prior['physical_macro_package_manifest_sha256'],
        logical_roots_and_protected_data_clock_masters_preserved=True,
        corrected_async_semantics_differ_from_legacy=True,extracted_sta=False,physical_frequency_certified=False,
        evidence={str(p):sha(p)for p in [proof,checkpoint,library,macros[0],
            *[p for p in out.iterdir()if p.is_file()]]})
    (out/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in correction.items()if k!='evidence'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--legacy-mapping',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--timeout',type=int,default=900)
    args=p.parse_args()
    if not 1<=args.timeout<=3600:p.error('timeout must be1..3600 seconds')
    run(args)
