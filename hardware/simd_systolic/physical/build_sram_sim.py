#!/usr/bin/env python3
"""Build co-simulation from the exact RTL snapshot using a generated SRAM.

The physical snapshot authenticates implementation identity; its frequency and
the macro's analog behavior remain separately qualified.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
from build import build
from runtime import validated_build
from simd_eval.architecture import ARCH
from sram_package import load_package


def run(args):
    snapshot=args.snapshot.resolve();identity=json.loads((snapshot/'build_identity.json').read_text())
    package=load_package(args.sram_package,ARCH)
    if identity.get('sram_package_manifest_sha256')!=package['manifest_sha256']:
        raise ValueError('physical RTL and requested SRAM package differ')
    digest=hashlib.sha256(package['wrapper'].read_bytes()).hexdigest()
    if identity['input_manifest'].get('Asap7SramMacro.sv')!=digest:
        raise ValueError('physical RTL wrapper differs from the verified macro model')
    if not(snapshot/'synthesis.json').is_file():raise ValueError('complete the implementation synthesis first')
    directory=build(kind=identity['scope'],stages=2,rows=identity['rows'],cols=identity['cols'],
        jobs=args.jobs,hierarchical=True,physical_snapshot=snapshot,label=args.label,
        sim_threads=args.sim_threads,cohort=args.cohort)
    guard,caps=validated_build(directory)
    caps.update(sram_native_macro=package['native'],sram_package_manifest_sha256=package['manifest_sha256'],
        storage_implementation_detail='generated ASAP7 6T research macro logical model',
        sram_liberty_kind='estimated',sram_physical_certified=False)
    path=directory/'capabilities.json';pending=directory/'capabilities.json.tmp'
    pending.write_text(json.dumps(caps,indent=2)+'\n');pending.replace(path)
    guard,caps=validated_build(directory)
    report=dict(kind='generated_sram_model_build',rtl_sha256=caps['rtl_sha256'],
        simulator_sha256=caps['simulator_sha256'],testbench_sha256=caps['testbench_sha256'],
        native_macro=package['native'],macro_package_manifest_sha256=package['manifest_sha256'],
        physical_snapshot=str(snapshot),physical_frequency_certified=False,
        transistor_behavior_verified=False)
    (directory/'sram_model_build_identity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(str(directory))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--sram-package',type=Path,required=True);p.add_argument('--cohort',type=Path)
    p.add_argument('--jobs',type=int,default=8);p.add_argument('--sim-threads',type=int,choices=[1,2,4,8],default=1)
    p.add_argument('--label',default='openfinram_model');run(p.parse_args())
