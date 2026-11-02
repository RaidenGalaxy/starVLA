#!/usr/bin/env python3
"""Continue a verified SRAM system floorplan through placement or dual CTS.

The source checkpoint and its separate physical macro revision are preserved.
Estimated SRAM timing and placement parasitics cannot certify frequency.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from asap7_platform import profile
from prepare_xyce import ROOT,sha
sys.path.insert(0,str(ROOT/'python'))
from sram_package import load_package
from simd_eval.architecture import ARCH


def run(args):
    parent=args.source.resolve();prior=json.loads((parent/'verification.json').read_text())
    if not prior.get('passed')or not prior.get('grid_generation_and_connectivity_passed')or not prior.get('all_protected_supplies_connected'):
        raise ValueError('requires verified protected supply repair of a full system grid')
    floor_path=Path(prior['source_verification']);floor=json.loads(floor_path.read_text())
    if sha(floor_path)!=prior['source_verification_sha256']:raise ValueError('parent floorplan changed')
    for record in(prior,floor):
        for path,digest in record['evidence'].items():
            if sha(path)!=digest:raise ValueError('parent evidence changed: '+path)
    macro=load_package(args.sram_package,ARCH)
    if macro['manifest_sha256']!=prior['physical_macro_package_manifest_sha256']:
        raise ValueError('selected physical macro differs from the database')
    technology=profile('TT');mapping=Path(floor['source_mapping'])
    identity=json.loads((mapping/'build_identity.json').read_text())
    from async_mapping_policy import validate as validate_async
    validate_async(mapping,identity)
    if identity['mapped_sha256']!=prior['source_mapped_sha256']or sha(technology['liberty'])!=identity['liberty_sha256']:
        raise ValueError('mapped identity or standard cell library changed')
    if floor['core_period_ns']!=1 or floor['memory_period_ns']!=2:raise ValueError('target clocks differ')
    checkpoint=parent/'repaired.odb';placed_from=args.placement_from.resolve()if args.placement_from else None
    if placed_from:
        resume=json.loads((placed_from/'verification.json').read_text())
        if resume['source_verification_sha256']!=sha(parent/'verification.json')or \
                resume['source_mapped_sha256']!=prior['source_mapped_sha256']or \
                resume['physical_macro_package_manifest_sha256']!=macro['manifest_sha256']:
            raise ValueError('placement checkpoint has another floorplan, mapping or macro')
        checkpoint=placed_from/'pre_electrical.odb'
        if not checkpoint.exists():checkpoint=placed_from/'post_global.odb'
        if resume['evidence'].get(str(checkpoint))!=sha(checkpoint):raise ValueError('placement checkpoint changed')
    out=args.out.resolve()
    if out.exists():raise ValueError('preserve prior system placement diagnostics')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'experiment_driver.py')
    original=floor_path.parent
    shutil.copy2(original/'cdc_constraints.tcl',out/'cdc_constraints.tcl')
    shutil.copy2(parent/'protected_power.tcl',out/'protected_power.tcl')
    shutil.copy2(ROOT/'physical/protected_ties.tcl',out/'protected_ties.tcl')
    shutil.copy2(ROOT/'physical/protected_clocks.tcl',out/'protected_clocks.tcl')
    (out/'design.sdc').write_text((original/'design.sdc').read_text().replace(str(original),str(out)))
    script=out/'place.tcl'
    script.write_text(f'''set_thread_count {args.jobs}
read_liberty {{{technology['liberty']}}}
read_liberty {{{macro['liberty']}}}
read_db {{{checkpoint}}}
set_cmd_units -time ns -capacitance fF
read_sdc {{{out/'design.sdc'}}}
source {{{technology['root']/'setRC.tcl'}}}
set_wire_rc -signal -layer M3
set_wire_rc -clock -layer M5
set_dont_use [get_lib_cells {{*/*x1p*_ASAP7* */*xp*_ASAP7* */SDF* */ICG*}}]
set_routing_layers -signal M2-M7 -clock M4-M7
source {{{out/'protected_power.tcl'}}}
'''+(''if placed_from else f'''
source {{{out/'protected_ties.tcl'}}}
simd_repair_protected_ties
buffer_ports
place_pins -hor_layers M4 -ver_layers M5
# Geometric search bounds virtual sizing cost; all paths are timed below.
global_placement -density 0.35 -initial_place_max_iter 20 -bin_grid_count 256 -routability_driven
write_db {{{out/'post_global.odb'}}}
''')+f'''
detailed_placement
write_db {{{out/'pre_electrical.odb'}}}
estimate_parasitics -placement
'''+(''if args.skip_electrical_repair else f'''
repair_design -cap_margin 20 -slew_margin 20
detailed_placement
write_db {{{out/'post_electrical.odb'}}}
''')+(f'''source {{{out/'protected_clocks.tcl'}}}
simd_protected_cts -root_buf {technology['cts_root']} -buf_list {{{' '.join(technology['cts_buffers'])}}}
set_propagated_clock [all_clocks]
detailed_placement
'''if args.stage=='cts'else'')+f'''global_connect
simd_connect_protected_supplies
estimate_parasitics -placement
check_placement -verbose
write_db {{{out/'placed.odb'}}}
write_verilog {{{out/'placed.v'}}}
report_clock_properties [all_clocks] > {{{out/'clocks.rpt'}}}
report_checks -path_delay max -group_count 10 -format full_clock_expanded -digits 9 > {{{out/'setup.rpt'}}}
report_checks -path_delay min -group_count 10 -format full_clock_expanded -digits 9 > {{{out/'hold.rpt'}}}
check_setup -verbose > {{{out/'coverage.rpt'}}}
puts SIMD_PLACED_SETUP
report_worst_slack -max -digits 9
puts SIMD_PLACED_HOLD
report_worst_slack -min -digits 9
report_design_area
set count 0
foreach inst [[ord::get_db_block] getInsts] {{
 if {{[[$inst getMaster] getName] eq "{macro['native']}"}} {{incr count}}
}}
if {{$count != 16}} {{error "SRAM instance count changed"}}
check_power_grid -net VDD -error_file {{{out/'pdn_VDD.rpt'}}}
check_power_grid -net VSS -error_file {{{out/'pdn_VSS.rpt'}}}
puts SIMD_PLACED_POWER_GRID_VERIFIED
''')
    local=ROOT/'.deps/tools/openroad';env=os.environ.copy()
    env['LD_LIBRARY_PATH']=':'.join([str(local/'opt/or-tools/lib'),str(local/'usr/lib/x86_64-linux-gnu'),env.get('LD_LIBRARY_PATH','')])
    log=out/'openroad.log';timed_out=False
    with log.open('w')as stream:
        process=subprocess.Popen([str(local/'usr/bin/openroad'),'-exit',str(script)],env=env,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        (out/'running.json').write_text(json.dumps(dict(pid=process.pid,timeout_seconds=args.timeout))+'\n')
        try:code=process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            code=process.wait()
    text=log.read_text();slacks={}
    for kind in('SETUP','HOLD'):
        match=re.search('SIMD_PLACED_'+kind+r'.*?worst slack\s+([-+\d.]+)',text,re.S)
        if match:slacks[kind.lower()+'_wns_ns']=float(match[1])
    passed=code==0 and not timed_out and 'SIMD_PLACED_POWER_GRID_VERIFIED'in text
    result=dict(kind='full_mapped_system_placement_diagnostic',passed=passed,returncode=code,
        timed_out=timed_out,timeout_seconds=args.timeout,stage=args.stage,
        source_verification=str(parent/'verification.json'),source_verification_sha256=sha(parent/'verification.json'),
        source_mapped_sha256=prior['source_mapped_sha256'],source_rtl_sha256=identity['rtl_sha256'],
        mapped_macro_package_manifest_sha256=floor['mapped_macro_package_manifest_sha256'],
        physical_macro_package_manifest_sha256=macro['manifest_sha256'],
        arithmetic_stages=2,core_period_ns=1,memory_period_ns=2,
        placement_resumed_from=str(placed_from)if placed_from else None,
        placement_source_checkpoint=str(checkpoint),
        electrical_repair_requested=not args.skip_electrical_repair,
        standard_cell_placement_complete=passed,dual_clock_cts_complete=passed and args.stage=='cts',
        power_grid_connectivity_verified='SIMD_PLACED_POWER_GRID_VERIFIED'in text,
        timing=slacks,timing_evidence='placement RC estimate; no signal routing or extraction',
        physical_frequency_certified=False,whole_macro_characterized=False,
        qualified_async_mapping_root=str(mapping),
        evidence={str(p):sha(p)for p in [parent/'verification.json',checkpoint,
            *([placed_from/'verification.json']if placed_from else[]),
            technology['liberty'],macro['liberty'],*[p for p in out.iterdir()if p.is_file()]]})
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='evidence'},indent=2))
    if not passed:raise RuntimeError('system placement diagnostic failed or timed out; evidence retained')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--sram-package',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['place','cts'],default='cts')
    p.add_argument('--placement-from',type=Path,help='resume an authenticated completed or failed diagnostic post_global.odb')
    p.add_argument('--skip-electrical-repair',action='store_true',help='CTS/placement diagnostic only; electrical repair and routed checks remain required for acceptance')
    p.add_argument('--timeout',type=int,default=3600);p.add_argument('--jobs',type=int,default=8);args=p.parse_args()
    if not 1<=args.jobs<=16 or not 1<=args.timeout<=7200:p.error('jobs 1..16, timeout 1..7200 required')
    run(args)
