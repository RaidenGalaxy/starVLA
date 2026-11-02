#!/usr/bin/env python3
"""Classify unannotated drivers from the authenticated routed database."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from run import ROOT,sha,tool


def run(args):
    source=args.directory.resolve();out=args.out.resolve()
    if out.exists():raise ValueError('preserve previous coverage audits')
    manifest=source/'canonical_export.json';parent=json.loads(manifest.read_text())
    if not parent['geometry_and_connectivity_preserved']:raise ValueError('geometry is not authenticated')
    for filename,digest in parent['evidence'].items():
        if sha(filename)!=digest:raise ValueError('export evidence changed: '+filename)
    from parasitics import unannotated_names
    log=source/'route.log';names=unannotated_names(log.read_text())
    out.mkdir(parents=True)
    shutil.copy2(__file__,out/'audit_driver.py')
    shutil.copy2(ROOT/'physical/parasitic_drivers.tcl',out/'parasitic_drivers.tcl')
    inputs=out/'drivers.hex';inputs.write_text(''.join(name.encode().hex()+'\n'for name in sorted(names)))
    database=source/'canonical.odb';spef=source/'design.spef'
    script=out/'audit.tcl';script.write_text(f'''read_db {{{database}}}
source {{{out/'parasitic_drivers.tcl'}}}
set input [open {{{inputs}}} r]
set output [open {{{out/'drivers.tsv'}}} w]
puts $output "driver_utf8_hex\tclassification\tsinks\trouted"
while {{[gets $input hex]>=0}} {{
 set name [encoding convertfrom utf-8 [binary decode hex $hex]]
 lassign [simd_audit_parasitic_driver $name] classification sinks routed
 puts $output "$hex\t$classification\t$sinks\t$routed"
}}
close $input
close $output
puts SIMD_PARASITIC_DRIVER_AUDIT_COMPLETE
''')
    local=ROOT/'.deps/tools/openroad';env={**os.environ,'LD_LIBRARY_PATH':
        str(local/'opt/or-tools/lib')+':'+str(local/'usr/lib/x86_64-linux-gnu')}
    with(out/'openroad.log').open('w')as stream:
        result=subprocess.run([tool('openroad'),'-exit',str(script)],env=env,
            stdout=stream,stderr=subprocess.STDOUT,timeout=600)
    rows=[]
    for line in (out/'drivers.tsv').read_text().splitlines()[1:]:
        name,classification,sinks,routed=line.split('\t')
        rows.append(dict(name=bytes.fromhex(name).decode(),classification=classification,sinks=int(sinks),routed=int(routed)))
    qualified={r['name']for r in rows if r['classification']in('unconnected_output','zero_fanout_no_routing')and r['sinks']==0 and r['routed']==0}
    passed=result.returncode==0 and len(rows)==len(names)and qualified==set(names)
    paths=[manifest,database,spef,log,*[p for p in out.iterdir()if p.is_file()]]
    report=dict(kind='actual_routed_zero_fanout_parasitic_audit',passed=passed,
        unannotated_drivers=len(names),proven_zero_fanout_drivers=len(qualified),
        nonqualifying=[r for r in rows if r['name']not in qualified],
        source_spef_sha256=sha(spef),no_timing_exceptions=True,
        driver_table=str(out/'drivers.tsv'),evidence={str(p):sha(p)for p in paths})
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k not in('evidence','nonqualifying')},indent=2))
    if not passed:raise ValueError('some missing parasitics belong to connected or unknown drivers')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
