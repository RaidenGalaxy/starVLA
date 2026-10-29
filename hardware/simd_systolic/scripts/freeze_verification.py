#!/usr/bin/env python3
"""Freeze/verify the source cohort before starting long independent builds."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_manifest():
    files=[]
    for folder in ('src/main','rtl','vendor','sim','python','scripts','tests','configs'):
        files.extend(p for p in (ROOT/folder).rglob('*')if p.is_file()and
                     '__pycache__'not in p.parts and p.suffix not in('.pyc','.so'))
    files.extend(ROOT/n for n in ('build.sbt','dependencies.lock.json','Makefile'))
    # These physical-policy helpers are imported by the functional unit tests.
    files.extend(ROOT/'physical'/n for n in ('coverage.py','system_constraints.py'))
    files.extend((ROOT/'vendor/common_cells').glob('*.svh'))
    return {str(p.relative_to(ROOT)):sha(p)for p in sorted(files)}


def freeze(out):
    if out.exists():raise ValueError('cohort already exists; do not overwrite frozen evidence')
    manifest=source_manifest()
    cohort=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    lock=json.loads((ROOT/'dependencies.lock.json').read_text())
    pdk={}
    for family in ('asap7','asap7_sram'):
        for name,record in lock[family]['files'].items():
            p=ROOT/'.deps'/family/name
            actual=sha(p)
            if actual!=record['sha256']:raise ValueError('PDK differs from lock: '+name)
            pdk[family+'/'+name]=actual
    versions={}
    for command in (['verilator','--version'],['yosys','-V']):
        run=subprocess.run(command,capture_output=True,text=True,check=True)
        versions[command[0]]=run.stdout.strip()
    physical_reference={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'physical').rglob('*')if p.is_file()and '__pycache__'not in p.parts}
    report=dict(schema_version=2,cohort_sha256=cohort,source_manifest=manifest,
                physical_reference_manifest=physical_reference,physical_reference_only=True,
                pdk_manifest=pdk,process='ASAP7 RVT TT predictive reference; SRAM not physically certified',
                tool_versions=versions,functional_threshold_percent=99,
                code_coverage_threshold_percent=None,full_plan_complete=False)
    out.parent.mkdir(parents=True,exist_ok=True)
    snapshot=out.parent/'sources'
    for relative,digest in manifest.items():
        content=(ROOT/relative).read_bytes()
        if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('source changed while freezing: '+relative)
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    for relative,digest in physical_reference.items():
        content=(ROOT/relative).read_bytes()
        if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('physical reference changed while freezing: '+relative)
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    report['source_snapshot_directory']=str(snapshot.resolve())
    out.write_text(json.dumps(report,indent=2)+'\n')
    return report


def verify(path):
    frozen=json.loads(path.read_text());actual=source_manifest()
    snapshot=Path(frozen.get('source_snapshot_directory',path.parent/'sources'))
    if not snapshot.is_dir():raise ValueError('frozen source content is missing; create a new cohort directory')
    for relative,digest in frozen['source_manifest'].items():
        if sha(snapshot/relative)!=digest:raise ValueError('frozen source content changed: '+relative)
    for relative,digest in frozen.get('physical_reference_manifest',{}).items():
        if sha(snapshot/relative)!=digest:raise ValueError('frozen physical reference changed: '+relative)
    if actual!=frozen['source_manifest']:
        changed=sorted(k for k in set(actual)|set(frozen['source_manifest'])if actual.get(k)!=frozen['source_manifest'].get(k))
        raise ValueError('source cohort changed: '+', '.join(changed[:12]))
    for name,digest in frozen['pdk_manifest'].items():
        if sha(ROOT/'.deps'/name)!=digest:raise ValueError('PDK changed: '+name)
    return frozen


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path)
    parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    report=verify(args.out)if args.verify else freeze(args.out)
    print(json.dumps(dict(cohort_sha256=report['cohort_sha256'],files=len(report['source_manifest']),verified=True)))
