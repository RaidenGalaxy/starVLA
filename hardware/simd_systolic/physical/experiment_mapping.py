#!/usr/bin/env python3
"""Remap an authenticated F2 FP64 elaboration without changing arithmetic RTL."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from run import ROOT,sha,ARCH


def run(args):
    parent=args.source.resolve();identity=json.loads((parent/'build_identity.json').read_text())
    if identity['scope']!='fp64'or identity['stages']!=2 or identity['analysis_period_ns']!=1:
        raise ValueError('requires a verified F2 FP64 component at1ns')
    if identity['process_corner']!='TT'or identity['architecture_sha256']!=ARCH['architecture_sha256']:
        raise ValueError('parent process corner/architecture differs')
    numerical=json.loads((parent/'rtl_verification.json').read_text())
    if not numerical['passed']or numerical['rtl_sha256']!=identity['rtl_sha256']:
        raise ValueError('parent has no matching numerical verification')
    for name,digest in identity['input_manifest'].items():
        if sha(parent/'inputs'/name)!=digest:raise ValueError('parent RTL changed: '+name)
    if sha(parent/'premap.il')!=identity['premap_sha256']:raise ValueError('parent elaboration changed')
    out=ROOT/('.build/physical/asap7_TT_fp64_2_1x1_1ns_'+args.label)
    if out.exists():raise ValueError('preserve previous mapping diagnostics')
    out.mkdir(parents=True);shutil.copytree(parent/'inputs',out/'inputs')
    for name in('premap.il','rtl_verification.json'):shutil.copy2(parent/name,out/name)
    shutil.copy2(__file__,out/'mapping_experiment_driver.py')
    identity.update(mapping_parent=str(parent),mapping_parent_identity_sha256=sha(parent/'build_identity.json'),
        mapping_experiment_driver_sha256=sha(__file__),mapping_profile='choices')
    (out/'build_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    command=[sys.executable,str(ROOT/'physical/run.py'),'--target','fp64','--stages','2',
        '--rows','1','--period','1','--label',args.label,'--remap-premap','--mapping-profile','choices','--cts-only']
    subprocess.run(command,cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'physical/check_gates.py'),str(out),'--count','1000'],cwd=ROOT,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--label',required=True);run(p.parse_args())
