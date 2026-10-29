#!/usr/bin/env python3
"""Connect protected supply pins in an authenticated system floorplan.

The parent supplies the actual grid connectivity check. This derived checkpoint
must retain byte-identical DEF geometry and all protected signal connections;
no placement, grid regeneration or timing acceptance is inferred.
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from prepare_analog import sha

ROOT=Path(__file__).resolve().parents[1]


def run(args):
    source=args.source.resolve();prior=json.loads((source/'verification.json').read_text())
    if not prior['grid_generation_and_connectivity_passed']or not prior['dual_clock_sdc_loaded']:
        raise ValueError('parent has no complete grid/clock evidence')
    for path,digest in prior['evidence'].items():
        if sha(path)!=digest:raise ValueError('parent floorplan changed: '+path)
    out=args.out.resolve()
    if out.exists():raise ValueError('preserve previous power diagnostics')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'experiment_driver.py')
    shutil.copy2(ROOT/'physical/protected_power.tcl',out/'protected_power.tcl')
    script=out/'repair.tcl';script.write_text(f'''read_db {{{source/'floorplan.odb'}}}
proc dump_protected {{path}} {{
 set stream [open $path w]
 foreach inst [[ord::get_db_block] getInsts] {{
  if {{![$inst isDoNotTouch]}} {{continue}}
  foreach pin [$inst getITerms] {{
   set net [$pin getNet]
   set name "NULL"
   if {{$net ne "NULL"}} {{set name [$net getName]}}
   puts $stream "[$inst getName]\t[[$pin getMTerm] getName]\t[[$pin getMTerm] getSigType]\t$name"
  }}
 }}
 close $stream
}}
dump_protected {{{out/'protected_before.tsv'}}}
write_def {{{out/'before.def'}}}
source {{{out/'protected_power.tcl'}}}
simd_connect_protected_supplies
dump_protected {{{out/'protected_after.tsv'}}}
write_def {{{out/'after.def'}}}
write_db {{{out/'repaired.odb'}}}
''')
    local=ROOT/'.deps/tools/openroad'
    env={**os.environ,'LD_LIBRARY_PATH':':'.join([str(local/'opt/or-tools/lib'),
        str(local/'usr/lib/x86_64-linux-gnu'),os.environ.get('LD_LIBRARY_PATH','')])}
    with(out/'openroad.log').open('w')as stream:
        result=subprocess.run([str(local/'usr/bin/openroad'),'-exit',str(script)],
            stdout=stream,stderr=subprocess.STDOUT,env=env)
    def pins(name):
        return {tuple(f[:3]):f[3]for line in(out/name).read_text().splitlines()if(f:=line.split('\t'))}
    before=pins('protected_before.tsv');after=pins('protected_after.tsv')
    same_pins=before.keys()==after.keys()
    supplies=[p for p in after if p[2]in('POWER','GROUND')]
    good_supplies=bool(supplies)and all(after[p]==('VDD'if p[2]=='POWER'else'VSS')for p in supplies)
    same_signals=same_pins and all(before[p]==after[p]for p in after if p[2]not in('POWER','GROUND'))
    same_def=sha(out/'before.def')==sha(out/'after.def')
    passed=result.returncode==0 and same_pins and good_supplies and same_signals and same_def
    record=dict(kind='protected_system_supply_connection_repair',passed=passed,returncode=result.returncode,
        source_verification=str(source/'verification.json'),source_verification_sha256=sha(source/'verification.json'),
        source_mapped_sha256=prior['source_mapped_sha256'],
        physical_macro_package_manifest_sha256=prior['physical_macro_package_manifest_sha256'],
        protected_instances=len({p[0]for p in after}),protected_pin_set_unchanged=same_pins,
        protected_supply_pins=len(supplies),new_supply_connections=sum(before.get(p)=='NULL'for p in supplies),
        all_protected_supplies_connected=good_supplies,protected_signal_connections_unchanged=same_signals,
        serialized_def_identical=same_def,grid_connectivity_inherited_from_authenticated_parent=passed,
        grid_generation_and_connectivity_passed=passed,
        standard_cell_placement_complete=False,physical_frequency_certified=False,
        evidence={str(p):sha(p)for p in [source/'verification.json',source/'floorplan.odb',
            *[p for p in out.iterdir()if p.is_file()]]})
    (out/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items()if k!='evidence'},indent=2))
    if not passed:raise RuntimeError('protected supplies or retained geometry/connectivity failed verification')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);run(p.parse_args())
