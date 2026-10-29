#!/usr/bin/env python3
"""Inspect a frozen CTS checkpoint without interrupting the routing job."""
import argparse,json,os,subprocess
from pathlib import Path
from run import ROOT,sha,tool

def inspect(directory):
    directory=directory.resolve();db=directory/'post_cts.odb';sdc=directory/'design.sdc'
    initial=sha(db);out=directory/'cts_diagnostic';out.mkdir(exist_ok=True)
    pdk=ROOT/'.deps/asap7';lib=pdk/'asap7_RVT_TT.lib'
    tcl=out/'inspect.tcl';tcl.write_text(f'''set_thread_count 2
read_liberty {lib}
read_db {db}
read_sdc {sdc}
set_wire_rc -signal -layer M3
set_wire_rc -clock -layer M5
set_propagated_clock [all_clocks]
estimate_parasitics -placement
report_checks -path_delay min -group_count 20 -format full_clock_expanded -fields {{slew capacitance input_pin net}} -digits 9 > {out/'hold.rpt'}
report_checks -path_delay max -group_count 20 -format full_clock_expanded -fields {{slew capacitance input_pin net}} -digits 9 > {out/'setup.rpt'}
''')
    env=os.environ.copy();local=ROOT/'.deps/tools/openroad'
    env['LD_LIBRARY_PATH']=str(local/'opt/or-tools/lib')+':'+str(local/'usr/lib/x86_64-linux-gnu')
    with(out/'sta.log').open('w')as log:subprocess.run([tool('openroad'),'-exit',str(tcl)],check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
    if sha(db)!=initial:raise RuntimeError('CTS checkpoint changed during inspection')
    result=dict(evidence_kind='cts_placement_estimate_only',checkpoint_sha256=initial,sdc_sha256=sha(sdc),
        liberty_sha256=sha(lib),post_route_certified=False,setup_report=str(out/'setup.rpt'),hold_report=str(out/'hold.rpt'))
    (out/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);inspect(parser.parse_args().directory)
