#!/usr/bin/env python3
"""Check an authenticated mapped system with an explicit macro physical revision.

The mapped source and its original macro identity are retained. This diagnostic
may select a newer physical macro only when its native ports and functional
wrapper match exactly; it does not relabel synthesis or prior functional tests.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from asap7_platform import profile
from mapped_system_constraints import write_system_sdc
from prepare_analog import sha
from sram_package import load_package,inventory
from simd_eval.architecture import ARCH

def run(args):
    source=args.mapping.resolve();identity=json.loads((source/'build_identity.json').read_text())
    if identity['scope']not in('system','gemm')or identity['stages']!=2:
        raise ValueError('requires a complete mapped F2 system')
    if identity['architecture_sha256']!=ARCH['architecture_sha256']or identity['process_corner']!='TT':
        raise ValueError('mapped architecture/corner differs')
    net=source/'synth.v'
    if sha(net)!=identity['mapped_sha256']:raise ValueError('mapped system netlist changed')
    original=load_package(args.mapped_sram_package,ARCH);selected=load_package(args.sram_package,ARCH)
    if original['manifest_sha256']!=identity['sram_package_manifest_sha256']:
        raise ValueError('original mapped macro identity differs')
    if sha(original['wrapper'])!=sha(selected['wrapper'])or original['native']!=selected['native']:
        raise ValueError('physical revision changes the native functional interface')
    pins=lambda p:set(re.findall(r'^\s*PIN\s+(\S+)',p.read_text(),re.M))
    if pins(original['lef'])!=pins(selected['lef']):raise ValueError('physical revision changes pin set')
    capacity=inventory(net,selected['native'])
    if not capacity['all_nominal_bits_retained']:raise ValueError('mapped system does not retain the complete 128KiB SRAM')
    technology=profile('TT')
    if sha(technology['liberty'])!=identity['liberty_sha256']:raise ValueError('standard-cell library differs')
    out=args.out.resolve()
    if out.exists():raise ValueError('preserve earlier system floorplan diagnostics')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'experiment_driver.py')
    for name in('sram_floorplan.tcl','cdc_constraints.tcl','protected_power.tcl'):shutil.copy2(ROOT/'physical'/name,out/name)
    pdn=technology['root']/'openRoad/pdn/grid_strategy-M1-M2-M5-M6.tcl'
    shutil.copy2(pdn,out/'pdn.tcl')
    prepared=args.prepared_from.resolve()if args.prepared_from else None
    if prepared:
        prior=json.loads((prepared/'verification.json').read_text())
        if not prior.get('dual_clock_sdc_loaded')or prior['source_mapped_sha256']!=sha(net)or \
                prior['physical_macro_package_manifest_sha256']!=selected['manifest_sha256']:
            raise ValueError('prepared constraints have another mapping, macro revision or no tool-read proof')
        for path,digest in prior['evidence'].items():
            if sha(path)!=digest:raise ValueError('prepared floorplan evidence changed: '+path)
        if sha(prepared/'cdc_constraints.tcl')!=sha(out/'cdc_constraints.tcl'):
            raise ValueError('prepared CDC helper differs from production')
        text=(prepared/'design.sdc').read_text().replace(str(prepared),str(out))
        (out/'design.sdc').write_text(text)
        shutil.copy2(prepared/'design.cdc.json',out/'design.cdc.json')
    else:write_system_sdc(net,out/'design.sdc',out/'cdc_constraints.tcl')
    script=out/'floorplan.tcl';script.write_text(f'''read_lef {{{technology['tech_lef']}}}
read_lef {{{technology['cell_lef']}}}
read_lef {{{selected['lef']}}}
read_liberty {{{technology['liberty']}}}
read_liberty {{{selected['liberty']}}}
set_cmd_units -time ns -capacitance fF
read_verilog {{{net}}}
link_design {identity['top']}
read_sdc {{{out/'design.sdc'}}}
puts "SIMD_SYSTEM_SDC_LOADED"
report_clock_properties [all_clocks] > {{{out/'clocks.rpt'}}}
write_sdc {{{out/'effective.sdc'}}}
initialize_floorplan -utilization 30 -aspect_ratio 1 -core_space 3 -site asap7sc7p5t
source {{{technology['tracks']}}}
source {{{out/'sram_floorplan.tcl'}}}
write_db {{{out/'floorplan.odb'}}}
write_def {{{out/'floorplan.def'}}}
report_design_area
''')
    local=ROOT/'.deps/tools/openroad';env=os.environ.copy()
    env.update(SIMD_OUT=str(out),SIMD_SRAM_MASTER=selected['native'],SIMD_PDN_SCRIPT=str(out/'pdn.tcl'))
    env['LD_LIBRARY_PATH']=':'.join([str(local/'opt/or-tools/lib'),str(local/'usr/lib/x86_64-linux-gnu'),env.get('LD_LIBRARY_PATH','')])
    log=out/'openroad.log'
    with log.open('w')as stream:
        result=subprocess.run([str(local/'usr/bin/openroad'),'-exit',str(script)],stdout=stream,stderr=subprocess.STDOUT,env=env)
    passed=result.returncode==0 and(out/'floorplan.odb').is_file()and(out/'floorplan.def').is_file()
    data=dict(kind='mapped_system_macro_floorplan_full_row_grid_diagnostic',
        grid_generation_and_connectivity_passed=passed,returncode=result.returncode,
        dual_clock_sdc_loaded='SIMD_SYSTEM_SDC_LOADED'in log.read_text(),
        prepared_constraints_from=str(prepared)if prepared else None,
        source_mapping=str(source),source_rtl_sha256=identity['rtl_sha256'],source_mapped_sha256=sha(net),
        mapped_macro_package_manifest_sha256=original['manifest_sha256'],
        physical_macro_package_manifest_sha256=selected['manifest_sha256'],native_functional_wrapper_unchanged=True,
        capacity=capacity,core_period_ns=1,memory_period_ns=2,
        grid_scope='Complete mapped system before standard-cell placement; platform M1/M2 row rails and M5/M6 grid, sixteen placed macros',
        standard_cell_placement_complete=False,physical_frequency_certified=False,full_macro_drc_lvs_verified=False,
        evidence={str(p):sha(p)for p in [source/'build_identity.json',net,selected['lef'],selected['liberty'],
            *[p for p in out.iterdir()if p.is_file()]]})
    (out/'verification.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items()if k!='evidence'},indent=2))
    if not passed:raise RuntimeError('system floorplan/PDN diagnostic failed; retain its actual reports')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mapping',type=Path,required=True)
    p.add_argument('--mapped-sram-package',type=Path,required=True);p.add_argument('--sram-package',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prepared-from',type=Path,help='reuse authenticated, tool-read constraints without reparsing a large mapping')
    run(p.parse_args())
