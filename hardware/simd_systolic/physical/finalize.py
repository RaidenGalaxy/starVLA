#!/usr/bin/env python3
"""Recover final reports from a completed routed DB without rerunning routing."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).parent))
from run import sha,tool,summarize
DRIVER_SOURCE=Path(__file__).read_bytes()


def finalize(out):
    out=out.resolve();identity=json.loads((out/'build_identity.json').read_text())
    (out/'finalization_driver.py').write_bytes(DRIVER_SOURCE)
    identity['finalization_driver_sha256']=sha(out/'finalization_driver.py')
    (out/'build_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    env=os.environ.copy();local=ROOT/'.deps/tools/openroad'
    env['LD_LIBRARY_PATH']=str(local/'opt/or-tools/lib')+':'+str(local/'usr/lib/x86_64-linux-gnu')
    commands=(ROOT/'physical/run.tcl').read_text().split('puts "SIMD_FINAL_SETUP"')[1]
    # Commands are kept on individual lines so Tcl redirection wrappers do not
    # split option lists while recovering a historical report.
    commands='puts "SIMD_FINAL_SETUP"'+commands.replace('$::env(SIMD_OUT)',str(out))
    macro_library=('read_liberty '+identity['sram_liberty_file']+'\n') if identity['scope'] in ('system','gemm') else ''
    tcl=out/'finalize.tcl';tcl.write_text(f'''read_liberty {identity['liberty_file']}
{macro_library}set_cmd_units -time ns -capacitance fF
read_db {out/'design.odb'}
read_sdc {out/'design.sdc'}
puts SIMD_SPEF_READ_BEGIN
read_spef {out/'design.spef'}
report_parasitic_annotation -report_unannotated
puts SIMD_SPEF_READ_END
set_propagated_clock [all_clocks]
'''+commands)
    proc=subprocess.run([tool('openroad'),'-exit',str(tcl)],check=True,text=True,capture_output=True,env=env)
    (out/'finalize.log').write_text(proc.stdout+proc.stderr)
    # Keep original routing history; replace only the old final-report tail.
    route=(out/'route.log').read_text().split('SIMD_FINAL_SETUP')[0]
    (out/'route.log').write_text(route+proc.stdout)
    report=summarize(out,identity,float(identity.get('analysis_period_ns',identity['mapping_period_ns'])))
    report['finalizer_sha256']=sha(tcl)
    (out/'timing.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);finalize(p.parse_args().directory)
