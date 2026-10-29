#!/usr/bin/env python3
"""Exercise the actual sixteen-macro floorplan/PDN before expensive routing."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'physical'),str(ROOT/'python')]
from asap7_platform import profile
from simd_eval.architecture import ARCH
from sram_package import load_package


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    package=load_package(args.sram_package,ARCH);technology=profile('TT')
    out=args.out.resolve()
    if out.exists():raise ValueError('preserve existing diagnostic; choose a new --out')
    out.mkdir(parents=True)
    driver=out/'probe_driver.py';shutil.copy2(__file__,driver)
    net=out/'probe.v';native=package['native']
    text='module SramPowerProbe(input clk,ce_n,we_n,oe_n,input [9:0] A,input [63:0] D,output [1023:0] Q);\n'
    for i in range(16):
        text+=native+f' storage_{i}(.clk(clk),.ce_n(ce_n),.we_n(we_n),.oe_n(oe_n),.sdel(4\'b0),.A(A),.D(D),.Q(Q[{64*i+63}:{64*i}]));\n'
    net.write_text(text+'endmodule\n')
    floorplan=out/'sram_floorplan.tcl';shutil.copy2(ROOT/'physical/sram_floorplan.tcl',floorplan)
    protected_power=out/'protected_power.tcl';shutil.copy2(ROOT/'physical/protected_power.tcl',protected_power)
    pdn=out/'pdn.tcl';reference_pdn=technology['root']/'openRoad/pdn/grid_strategy-M1-M2-M5-M6.tcl'
    # This probe contains sixteen macros and no standard cells. Keep the
    # reference M5/M6 grid and macro connections; omit unused M1/M2 row rails
    # and their dense via arrays. Production system builds retain those rails.
    rows=reference_pdn.read_text().splitlines()
    pdn.write_text('\n'.join(row for row in rows if not any(token in row for token in (
        '-layer {M1}','-layer {M2}','-layers {M1 M2}','-layers {M2 M5}')))+'\n')
    script=out/'probe.tcl'
    script.write_text(f'''read_lef {{{technology['tech_lef']}}}
read_lef {{{technology['cell_lef']}}}
read_lef {{{package['lef']}}}
read_liberty {{{technology['liberty']}}}
read_liberty {{{package['liberty']}}}
read_verilog {{{net}}}
link_design SramPowerProbe
initialize_floorplan -utilization 30 -aspect_ratio 1 -core_space 3 -site asap7sc7p5t
source {{{technology['tracks']}}}
source {{{floorplan}}}
write_db {{{out/'floorplan.odb'}}}
''')
    env=os.environ.copy();local=ROOT/'.deps/tools/openroad'
    env.update(SIMD_OUT=str(out),SIMD_PDN_SCRIPT=str(pdn),SIMD_SRAM_MASTER=native)
    env['LD_LIBRARY_PATH']=':'.join([str(local/'opt/or-tools/lib'),str(local/'usr/lib/x86_64-linux-gnu'),env.get('LD_LIBRARY_PATH','')])
    log=out/'openroad.log'
    with log.open('w') as stream:
        result=subprocess.run([str(local/'usr/bin/openroad'),'-exit',str(script)],
            stdout=stream,stderr=subprocess.STDOUT,env=env)
    reports=[out/'pdn_VDD.rpt',out/'pdn_VSS.rpt',out/'pdn_failed_vias.rpt']
    # An absent report after early failure cannot become a clean check.
    connected=result.returncode==0 and (out/'floorplan.odb').is_file()
    data=dict(kind='sixteen_macro_power_floorplan_diagnostic',macro_instances=16,
        macro_package_manifest_sha256=package['manifest_sha256'],
        process='asap7',returncode=result.returncode,grid_generation_and_connectivity_passed=connected,
        physical_frequency_certified=False,full_macro_drc_lvs_verified=False,
        grid_scope='M5/M6 supply grid and sixteen macro PG connections; no standard cells or row rails',
        reference_pdn_sha256=sha(reference_pdn),
        evidence={p.name:dict(path=str(p),sha256=sha(p)) for p in
            [net,floorplan,protected_power,pdn,script,log,*reports,driver] if p.is_file()})
    (out/'verification.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k!='evidence'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sram-package',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
