#!/usr/bin/env python3
"""Export a corrected system without top-port bus/net name collisions."""
import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from run import ROOT,sha,tool
from asap7_platform import profile
from async_mapping_policy import validate_checkpoint


def run(args):
    source=args.source.resolve();proof=source/'verification.json';prior=json.loads(proof.read_text())
    validate_checkpoint(source,prior)
    for name,digest in prior['evidence'].items():
        if sha(name)!=digest:raise ValueError('system source evidence changed: '+name)
    out=args.out.resolve()
    if out.exists():raise ValueError('preserve previous export attempts')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'export_driver.py')
    for name in('canonical_net_names.tcl','canonical_port_aliases.tcl'):
        shutil.copy2(ROOT/'physical'/name,out/name)
    for name in('cdc_constraints.tcl','protected_power.tcl','reset_repair.tcl'):
        shutil.copy2(source/name,out/name)
    (out/'design.sdc').write_text((source/'design.sdc').read_text().replace(str(source),str(out)))
    library=profile('TT')['liberty']
    if prior['evidence'].get(str(library))!=sha(library):raise ValueError('system export process library differs')
    macros=[Path(p)for p in prior['evidence']if p.endswith('.lib')and Path(p)!=library]
    if len(macros)!=1:raise ValueError('system export must bind one SRAM library')
    script=out/'export.tcl';script.write_text(f'''read_liberty {{{library}}}
read_liberty {{{macros[0]}}}
read_db {{{source/'repaired.odb'}}}
source {{{out/'canonical_net_names.tcl'}}}
source {{{out/'canonical_port_aliases.tcl'}}}
simd_export_port_aliases {{{out}}}
''')
    local=ROOT/'.deps/tools/openroad'
    env={**os.environ,'LD_LIBRARY_PATH':':'.join([str(local/'opt/or-tools/lib'),
        str(local/'usr/lib/x86_64-linux-gnu'),os.environ.get('LD_LIBRARY_PATH','')])}
    with(out/'openroad.log').open('w')as stream:
        result=subprocess.run([tool('openroad'),'-exit',str(script)],stdout=stream,
            stderr=subprocess.STDOUT,env=env,timeout=args.timeout)
    match=re.search(r'SIMD_PORT_SHADOWING_NETS_EXPORTED (\d+)',(out/'openroad.log').read_text())
    if result.returncode or not match:raise RuntimeError('port export failed; evidence retained')
    if sha(out/'before.def')!=sha(out/'restored.def'):raise ValueError('port export changed restored full DEF')
    if len({sha(out/name)for name in('before_connectivity.tsv','export_connectivity.tsv','restored_connectivity.tsv')})!=1:
        raise ValueError('port export changed connections, ports, geometry or protection')
    record=dict(kind='same_system_port_net_shadowing_export_correction',passed=True,
        port_shadowing_nets_renamed=int(match[1]),all_connections_ports_geometry_protection_preserved=True,
        full_def_roundtrip_identical=True,core_period_ns=1,memory_period_ns=2,
        source_verification=str(proof),source_verification_sha256=sha(proof),
        source_rtl_sha256=prior['source_rtl_sha256'],source_mapped_sha256=prior['source_mapped_sha256'],
        physical_macro_package_manifest_sha256=prior['physical_macro_package_manifest_sha256'],
        logical_roots_and_protected_data_clock_masters_preserved=True,physical_frequency_certified=False,
        **{k:prior[k]for k in('qualified_async_mapping_root','async_mapping_correction_verified',
            'async_correction_verification','async_correction_verification_sha256')if k in prior},
        evidence={str(p):sha(p)for p in [proof,source/'repaired.odb',library,macros[0],
            *[p for p in out.iterdir()if p.is_file()]]})
    (out/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items()if k!='evidence'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--timeout',type=int,default=1200)
    a=p.parse_args()
    if not 1<=a.timeout<=3600:p.error('timeout1..3600seconds')
    run(a)
