#!/usr/bin/env python3
"""Report actual Verilator counters, separately from functional/STA coverage."""
import argparse
import hashlib
import json
from pathlib import Path


def parse(path):
    records=[]
    for line in Path(path).read_text().splitlines():
        if not line.startswith("C '"):continue
        metadata,number=line[3:].rsplit("' ",1)
        attrs={}
        for part in metadata.split('\x01'):
            if '\x02'in part:
                key,value=part.split('\x02',1);attrs[key]=value
        count=int(number)
        if count<0:raise ValueError('negative coverage counter')
        records.append((metadata,attrs,count))
    if not records:raise ValueError('no actual Verilator coverage counters: '+str(path))
    return records


def summarize(paths,identity=None):
    merged={};files=[]
    for path in paths:
        path=Path(path);records=parse(path)
        if identity is not None:
            proof=json.loads(path.with_name(path.name+'.identity.json').read_text())
            if any(proof.get(k)!=identity.get(k)for k in ('rtl_sha256','simulator_sha256'))or not proof.get('functional_run_passed'):
                raise ValueError('code coverage belongs to another or failed build')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=proof['coverage_sha256']:raise ValueError('code coverage changed')
        files.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for key,attrs,count in records:
            if key not in merged:merged[key]=[attrs,0]
            merged[key][1]+=count
    groups={};unhit=[]
    for key,(attrs,count)in merged.items():
        page=attrs.get('page','');kind=next((name for name in('line','toggle','expr','branch','fsm')if name in page),attrs.get('t',attrs.get('type','unknown')))
        filename=attrs.get('f',attrs.get('filename','unknown'))
        owner=('third_party_arithmetic'if Path(filename).name.startswith('fpnew_')or Path(filename).name=='lzc.sv'else
               'modified_floating_arithmetic'if Path(filename).name.startswith('SimdFp')else'project_rtl')
        group=groups.setdefault(owner+'/'+kind,dict(total_points=0,hit_points=0))
        group['total_points']+=1;group['hit_points']+=int(count>0)
        if count==0:unhit.append(attrs)
    for group in groups.values():group['percent']=100*group['hit_points']/group['total_points']
    return dict(scope='observed_verilator_code_coverage',inputs=files,groups=groups,
                unhit_points=unhit,threshold_percent=None,
                limitations=['Verilator counter model; expression coverage is not a claim of MC/DC',
                             'Signals excluded by Verilator width limits and constant optimization are not counted',
                             'Equivalent instance counters are merged by Verilator; topology functional bins remain per PE',
                             'Coverage is not a correctness probability; timing/CDC physical signoff is separate'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('inputs',nargs='+',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--build-dir',required=True,type=Path);args=parser.parse_args()
    from runtime import validated_build
    guard,caps=validated_build(args.build_dir)
    if not caps.get('code_coverage_instrumented'):raise ValueError('a coverage-instrumented build is required')
    report=summarize(args.inputs,caps);report['identity']={k:caps[k]for k in('rtl_sha256','simulator_sha256','target','stages','rows','cols')}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k!='unhit_points'},indent=2))
