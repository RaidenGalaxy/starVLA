#!/usr/bin/env python3
"""Audit mapped SRAM capacity and generated CDC constraints, without signoff."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'physical'), str(ROOT/'python')]
from mapped_system_constraints import crossings, system_sdc, write_system_sdc
from simd_eval.architecture import ARCH
from simd_eval.constraints import system_policy
from sram_package import load_package, inventory


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def check(directory, package_directory, output):
    directory = directory.resolve()
    identity = json.loads((directory/'build_identity.json').read_text())
    if (identity['scope'] not in ('system','gemm') or identity['stages'] != 2 or
            identity.get('architecture_sha256') != ARCH['architecture_sha256']):
        raise ValueError('expected current F2 integrated mapping')
    net = directory/'synth.v'
    if sha(net) != identity['mapped_sha256']:
        raise ValueError('mapped netlist changed')
    from async_mapping_policy import validate as validate_async
    validate_async(directory,identity)
    for name,digest in identity['input_manifest'].items():
        if sha(directory/'inputs'/name) != digest:
            raise ValueError('mapped source changed: '+name)
    package = load_package(package_directory, ARCH)
    if identity['sram_package_manifest_sha256'] != package['manifest_sha256']:
        raise ValueError('mapping uses a different macro package')
    counts = inventory(net, package['native'])
    if not counts['all_nominal_bits_retained']:
        raise ValueError('128KiB macro inventory incomplete')
    # These are diagnostic snapshots; they do not overwrite production SDCs.
    helper = directory/'diagnostic_cdc_constraints.tcl'
    shutil.copy2(ROOT/'physical/cdc_constraints.tcl', helper)
    sdc = directory/'diagnostic.sdc'
    write_system_sdc(net, sdc, helper)
    records = crossings(net)
    policy = system_policy(sdc.read_text(), system_sdc(net, helper))
    policy.update(gray_groups=len({(r['fifo'],r['direction']) for r in records}),
        fifo_count=len({r['fifo'] for r in records}), gray_bits=len(records),
        physical_signoff=False)
    if not policy['clean']:
        raise ValueError('generated constraint audit failed')
    constraints = directory/'diagnostic_constraint_analysis.json'
    constraints.write_text(json.dumps(policy, indent=2)+'\n')
    synthesis = json.loads((directory/'synthesis.json').read_text())
    report = {key:synthesis[key] for key in ('scope','stages','rows','cols','process',
        'process_corner','core_period_ps','memory_period_ps','rtl_sha256',
        'sram_macro_name','sram_package_manifest_sha256','standard_cell_area_um2',
        'sram_macro_area_um2','mapped_cell_and_macro_area_um2')}
    report.update(evidence_kind='generated_6t_sram_system_synthesis',
        macro_counts=counts['macro_counts'], capacity_bytes=sum(counts['retained_bytes'].values()),
        physical_frequency_certified=False, sram_physical_certified=False, cdc_summary=policy,
        evidence={name:dict(path=str(path),sha256=sha(path)) for name,path in {
            'synthesis':directory/'synthesis.json','identity':directory/'build_identity.json',
            'netlist':net, 'inventory':directory/'memory_inventory.json',
            'sdc':sdc,'cdc':sdc.with_suffix('.cdc.json'),'constraints':constraints,
            'cdc_helper':helper,'constraint_generator':ROOT/'physical/mapped_system_constraints.py',
            'audit_script':Path(__file__)}.items()})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='evidence'},indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--sram-package', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    check(args.directory,args.sram_package,args.out)
