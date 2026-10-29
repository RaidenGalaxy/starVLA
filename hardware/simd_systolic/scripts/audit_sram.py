#!/usr/bin/env python3
"""Inspect actual SRAM views and verification evidence; fail closed on gaps."""
import argparse
import hashlib
import json
import re
import math
import shutil
import struct
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from simd_eval.architecture import ARCH


def gds_structures(path):
    data=Path(path).read_bytes();offset=0;names=[];layers=set()
    while offset+4<=len(data):
        length,kind,_=struct.unpack('>HBB',data[offset:offset+4])
        if length==0:break  # legal block padding after ENDLIB
        if length<4 or offset+length>len(data):raise ValueError('invalid GDS record')
        payload=data[offset+4:offset+length];offset+=length
        if kind==6:names.append(payload.decode().rstrip('\0'))
        if kind==13:layers.add(struct.unpack('>h',payload)[0])
    return names,sorted(layers)


def audit(root=None):
    root=Path(root or ROOT/'.deps/asap7_sram');macro=ARCH['sram_macro']
    files=[p for p in root.rglob('*')if p.is_file() and p.suffix!='.part']
    identities={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files}
    structures={};layers={}
    for p in files:
        if p.suffix=='.gds':structures[str(p.relative_to(root))],layers[str(p.relative_to(root))]=gds_structures(p)
    gaps=[]
    if not any(macro in names for names in structures.values()):gaps.append('No GDS structure matches the selected 1024x64 macro cell')
    if not any(p.suffix in ('.sp','.spice','.cdl')for p in files):gaps.append('Transistor netlist and LVS reference are missing')
    evidence_path=root/'physical_evidence.json'
    verified={};corner_libraries={}
    if evidence_path.exists():
        evidence=json.loads(evidence_path.read_text())
        for key in ('drc','lvs','pex','characterization_TT','characterization_SS','characterization_FF'):
            record=evidence.get(key,{})
            target=(root/record.get('file','missing')).resolve()
            valid=(root.resolve() in target.parents and target.is_file() and record.get('passed') is True and
                   hashlib.sha256(target.read_bytes()).hexdigest()==record.get('sha256') and
                   evidence.get('macro')==macro and evidence.get('views_sha256')==identities_without_evidence(identities))
            verified[key]=valid
        for corner,voltage,temperature in [('TT',.70,0),('SS',.63,100),('FF',.77,25)]:
            record=evidence.get('characterization_'+corner,{})
            library=(root/record.get('liberty_file','missing')).resolve()
            valid=root.resolve() in library.parents and library.is_file()
            raw=library.read_text()if valid else ''
            def number(name):
                match=re.search(r'\b'+name+r'\s*:\s*([-+\d.eE]+)\s*;',raw)
                return float(match[1])if match else float('nan')
            valid=valid and re.search(r'\bcell\s*\(\s*"?'+re.escape(macro)+r'"?\s*\)',raw) is not None
            valid=valid and math.isclose(number('nom_voltage'),voltage) and math.isclose(number('nom_temperature'),temperature)
            valid=valid and record.get('verified_period_ps')==ARCH['memory_period_ps']
            verified['characterization_'+corner]=verified.get('characterization_'+corner,False) and valid
            if verified['characterization_'+corner]:corner_libraries[corner]=str(library)
    for key in ('drc','lvs','pex','characterization_TT','characterization_SS','characterization_FF'):
        if not verified.get(key):gaps.append('Missing authenticated '+key+' evidence')
    return dict(macro=macro,architecture_sha256=ARCH['architecture_sha256'],
        physical_certified=not gaps,functional_model_only=bool(gaps),gaps=gaps,
        corner_libraries=corner_libraries,
        views_sha256=identities,gds_structures=structures,gds_layers=layers,
        tools={name:shutil.which(name)for name in ('ngspice','calibre','klayout')},
        capacity_bytes=ARCH['sram_capacity_bytes'],instances=ARCH['sram_macro_count'],
        memory_period_ps=ARCH['memory_period_ps'])


def identities_without_evidence(identities):
    # Only actual design views participate; reports authenticate those views.
    return {name:digest for name,digest in identities.items()if Path(name).suffix.lower() in
            ('.gds','.lef','.lib','.v','.sv','.sp','.spice','.cdl')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path)
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/migration_asap7_f2/sram_audit.json')
    p.add_argument('--require-certified',action='store_true');args=p.parse_args()
    report=audit(args.root);args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k]for k in ('macro','physical_certified','gaps','tools')},indent=2))
    if args.require_certified and not report['physical_certified']:sys.exit(1)
