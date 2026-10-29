#!/usr/bin/env python3
"""Audit the incoming-context to ready path in the actual mapped PE."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from run import ROOT,sha,tool
sys.path.insert(0,str(ROOT/'python'))
from simd_eval.constraints import core_policy


def check(directory):
    directory=directory.resolve()
    identity=json.loads((directory/'build_identity.json').read_text())
    caps=json.loads((directory/'generated/capabilities.json').read_text())
    if identity['scope']!='pe'or identity['stages']<=2:
        raise ValueError('an HF PE physical build is required')
    net=directory/'synth.v';sdc=directory/'design.sdc'
    lib=Path(identity['liberty_file'])
    if sha(net)!=identity['mapped_sha256']or sha(lib)!=identity['liberty_sha256']:
        raise RuntimeError('physical netlist or process library changed')
    if not core_policy(sdc.read_text(),identity['analysis_period_ns'])['clean']:
        raise ValueError('complete core constraints required for path audit')
    script=directory/'admission_cone.tcl'
    script.write_text(f'''read_liberty {lib}
read_verilog {net}
link_design SimdPE
read_sdc {sdc}
set contexts [get_ports io_in_bits_ctx*]
set ready [get_ports io_in_ready]
if {{[llength $contexts]!=4 || [llength $ready]!=1}} {{error "missing PE interface ports"}}
set paths [find_timing_paths -from $contexts -to $ready -path_delay max -group_count 1]
puts "SIMD_CONTEXT_READY_PATHS [llength $paths]"
report_checks -from $contexts -to $ready -path_delay max -group_count 1 > {directory/'admission_cone.rpt'}
exit
''')
    env=os.environ.copy();local=ROOT/'.deps/tools/openroad'
    env['LD_LIBRARY_PATH']=str(local/'opt/or-tools/lib')+':'+str(local/'usr/lib/x86_64-linux-gnu')
    proc=subprocess.run([tool('sta'),'-exit',str(script)],capture_output=True,text=True,env=env,timeout=300)
    (directory/'admission_cone.log').write_text(proc.stdout+proc.stderr)
    marker=re.search(r'^SIMD_CONTEXT_READY_PATHS (\d+)$',proc.stdout,re.M)
    if proc.returncode or marker is None:raise RuntimeError('mapped admission audit failed; see admission_cone.log')
    count=int(marker[1]);registered=caps.get('registered_context_admission',False)
    report=dict(passed=(count==0 if registered else count>0),scope='mapped_context_ready_dependency',
        registered_context_admission=registered,context_input_ports_checked=4,combinational_paths_returned=count,
        rtl_sha256=identity['rtl_sha256'],netlist_sha256=sha(net),liberty_sha256=sha(lib),sdc_sha256=sha(sdc),
        script_sha256=sha(script),path_report_sha256=sha(directory/'admission_cone.rpt'),
        timing_validated=False,limitation='Only audits removal/presence of this combinational dependency; extracted setup/hold remains required.')
    (directory/'verification_admission_cone.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    if not report['passed']:raise AssertionError('mapped context/ready dependency differs from admission contract')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    check(parser.parse_args().directory)
