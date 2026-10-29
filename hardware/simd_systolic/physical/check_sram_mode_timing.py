#!/usr/bin/env python3
"""Extracted controller STA for the SRAM wrapper's hardwired sdel=0 mode.

This scopes a real constant configuration, not a whole-macro certificate.
Dynamic input setup/hold checks remain; the four constant configuration flops
are identified from actual buffer/inverter connectivity and recorded explicitly.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def configuration_flops(text):
    canonical=lambda n:re.sub(r'[\\\s]','',n)
    cells=[];drivers={}
    for match in re.finditer(r'\b(\w+)\s+(\\\S+|[\w$]+)\s*\((.*?)\);',text,re.S):
        ports={p:canonical(n)for p,n in re.findall(r'\.(\w+)\s*\(([^()]*)\)',match[3])}
        typ,inst=match[1],match[2].lstrip('\\');cells.append((typ,inst,ports))
        if 'Y'in ports and 'A'in ports and len(ports)==2:drivers[ports['Y']]=(typ,ports['A'])
    def resolve(net,polarity=0,visited=None):
        visited=set() if visited is None else visited
        match=re.fullmatch(r'sdel\[(\d)\]',net)
        if match:return int(match[1]),polarity
        if net in visited or net not in drivers:return None
        visited.add(net);typ,source=drivers[net]
        if not typ.startswith(('INV','CKINV','BUF','HB')):return None
        return resolve(source,polarity^int(typ.startswith(('INV','CKINV'))),visited)
    result=[]
    for typ,inst,ports in cells:
        if 'DFF'not in typ or 'D'not in ports:continue
        resolved=resolve(ports['D'])
        if resolved is None:continue
        bit,polarity=resolved
        for port in ('Q','QN'):
            if port in ports:result.append(dict(bit=bit,pin=inst+'/'+port,
                case_value=polarity^int(port=='QN'),input_inversion=polarity,cell=typ))
    if sorted(r['bit']for r in result)!=[0,1,2,3]:raise ValueError('expected exactly four direct configuration flops')
    return sorted(result,key=lambda r:r['bit'])


def mode_coverage(raw,records):
    constant={r['pin'].rsplit('/',1)[0]+'/D'for r in records}
    endpoints=set(re.findall(r'^\s+(\S+/D)\s*$',raw,re.M))
    pattern=r'Warning:\s+There (?:are|is)\s+(\d+) unconstrained endpoints?\.'
    count=sum(map(int,re.findall(pattern,raw)))
    other=[line for line in raw.splitlines()if line.startswith('Warning:')and not re.fullmatch(pattern,line)]
    unexpected=sorted(endpoints-constant)
    return dict(unconstrained_endpoints=sorted(endpoints),hardwired_configuration_endpoints=sorted(constant),
        unexpected_unconstrained_endpoints=unexpected,other_warnings=other,
        reported_endpoint_count=count,dynamic_endpoints_covered=not unexpected and not other and count==len(endpoints))


def run(args):
    source=args.controller.resolve();out=args.out.resolve()
    if out.exists():raise ValueError('preserve existing timing evidence; choose a new --out')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'timing_driver.py')
    netlist=source/'netlist_for_lvs.v'
    odb=source/'ctrl_decode.odb'
    if not netlist.exists():netlist=source/'controller_eco.v'
    if not odb.exists():odb=source/'controller_eco.odb'
    constraints=args.constraints.resolve()if args.constraints else source/'ctrl_decode.sdc'
    records=configuration_flops(netlist.read_text())
    wrapper=ROOT/'artifacts/migration_asap7_f2/sram_openfinram/Asap7SramMacro.sv'
    if '.sdel(4\'b0)'not in wrapper.read_text():raise ValueError('wrapper is not statically configured to zero')
    technology=ROOT/'.deps/openfinram/tech'
    if args.library_profile=='compiler_tt25':
        if args.corner!='TT':raise ValueError('compiler library profile contains TT only')
        libs=sorted((technology/'lib').glob('*_RVT_TT.lib'));temperature=25
        if len(libs)!=5:raise ValueError('controller library set incomplete')
    else:
        sys.path.insert(0,str(ROOT/'physical'))
        from asap7_platform import profile
        libs=[profile(args.corner)['liberty']]
    from pvt import operating_point
    points=[operating_point(library)for library in libs]
    if any(point['problems']for point in points)or any(point['values']!=points[0]['values']for point in points):
        raise ValueError('missing or inconsistent controller Liberty operating points')
    voltage=points[0]['values']['nom_voltage'];temperature=points[0]['values']['nom_temperature']
    if args.library_profile=='compiler_tt25'and(voltage!=.7 or temperature!=25):
        raise ValueError('unexpected compiler TT operating point')
    script=out/'mode.tcl';rcx=ROOT/'.deps/asap7/rcx_patterns.rules'
    text=''.join('read_liberty {'+str(p)+'}\n'for p in libs)
    text+=f'''set_cmd_units -time ns -capacitance pF
read_db {{{odb}}}
read_sdc {{{constraints}}}
source {{{technology/'setRC.tcl'}}}
define_process_corner -ext_model_index 0 X
extract_parasitics -ext_model_file {{{rcx}}}
write_spef {{{out/'controller.spef'}}}
read_spef {{{out/'controller.spef'}}}
puts "MODE_ALL_SETUP"
report_worst_slack -max -digits 9
puts "MODE_ALL_HOLD"
report_worst_slack -min -digits 9
'''
    for r in records:
        text+=f"set_case_analysis 0 [get_ports {{sdel[{r['bit']}]}}]\n"
        text+=f"set_case_analysis {r['case_value']} [get_pins {{{r['pin']}}}]\n"
    if args.repair_hold:
        text+=f'''set_dont_touch [get_cells -hierarchical -quiet {{physical_delay_*}}]
remove_fillers
# Recreate the router's internal state after read_db before resizer callbacks
# mark nets dirty. A routed ODB alone does not initialize that in-memory state.
global_route -congestion_iterations 50
global_route -start_incremental
repair_timing -hold -hold_margin 0.010 -max_passes 12 -max_buffer_percent 20
detailed_placement
# The imported ODB still contains the old detailed wires. Their pin endpoints
# no longer match nets split by the hold buffers; rebuild signal/clock wires
# from the updated guides rather than mixing old and new connectivity.
foreach net [[ord::get_db_block] getNets] {{
  if {{[$net getSigType] eq "SIGNAL" || [$net getSigType] eq "CLOCK"}} {{
    set wire [$net getWire]
    if {{$wire ne "NULL"}} {{odb::dbWire_destroy $wire}}
  }}
}}
global_route -end_incremental
write_guides {{{out/'route.guide'}}}
detailed_route -output_drc {{{out/'route.drc'}}}
extract_parasitics -ext_model_file {{{rcx}}}
write_spef {{{out/'controller_eco.spef'}}}
read_spef {{{out/'controller_eco.spef'}}}
write_verilog {{{out/'controller_eco.v'}}}
write_def {{{out/'controller_eco.def'}}}
write_db {{{out/'controller_eco.odb'}}}
'''
    text+=f'''puts "MODE_FIXED_SETUP"
report_worst_slack -max -digits 9
puts "MODE_FIXED_HOLD"
report_worst_slack -min -digits 9
report_checks -path_delay max -group_count 10 -digits 9 > {{{out/'setup.rpt'}}}
report_checks -path_delay min -group_count 10 -digits 9 > {{{out/'hold.rpt'}}}
report_check_types -max_slew -max_capacitance -max_fanout -min_period -min_pulse_width -violators > {{{out/'electrical.rpt'}}}
check_setup -verbose > {{{out/'coverage.rpt'}}}
'''
    script.write_text(text)
    env=os.environ.copy();local=ROOT/'.deps/tools/openroad'
    env['LD_LIBRARY_PATH']=':'.join([str(local/'opt/or-tools/lib'),str(local/'usr/lib/x86_64-linux-gnu'),env.get('LD_LIBRARY_PATH','')])
    with(out/'openroad.log').open('w')as log:
        result=subprocess.run([str(local/'usr/bin/openroad'),'-exit',str(script)],env=env,stdout=log,stderr=subprocess.STDOUT)
    raw=(out/'openroad.log').read_text()
    def slack(marker):
        match=re.search(marker+r'\s+worst slack\s+([-\d.]+)',raw)
        return float(match[1])if match else None
    coverage=(out/'coverage.rpt').read_text()if(out/'coverage.rpt').exists()else ''
    coverage_result=mode_coverage(coverage,records)
    coverage_result['dynamic_endpoints_covered']&=(out/'coverage.rpt').exists()
    report=dict(kind='extracted_sram_controller_fixed_mode_diagnostic',period_ns=2,
        library_profile='compiler_tt25'if args.library_profile=='compiler_tt25'else'architecture',process_corner=args.corner,
        pvt_voltage_V=voltage,pvt_temperature_C=temperature,operating_points=points,
        hold_eco_applied=args.repair_hold,source_checkpoint_is_eco=odb.name=='controller_eco.odb',
        controller_only_evidence=True,
        coverage=coverage_result,
        mode_sdel=0,mode_applies_after_configuration_capture=True,configuration_flops=records,
        all_modes_setup_wns_ns=slack('MODE_ALL_SETUP'),all_modes_hold_wns_ns=slack('MODE_ALL_HOLD'),
        fixed_mode_setup_wns_ns=slack('MODE_FIXED_SETUP'),fixed_mode_hold_wns_ns=slack('MODE_FIXED_HOLD'),
        returncode=result.returncode,physical_frequency_certified=False,whole_macro_characterized=False,
        evidence={str(p):sha(p)for p in [odb,constraints,
            netlist,technology/'setRC.tcl',rcx,wrapper,*libs,
            *[p for p in out.iterdir()if p.is_file()]]})
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='evidence'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--controller',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repair-hold',action='store_true',help='separate routed controller ECO candidate; does not replace the selected macro')
    p.add_argument('--library-profile',choices=['compiler_tt25','architecture','architecture_tt25'],default='compiler_tt25',
        help='architecture_tt25 is a legacy alias; architecture uses the selected --corner')
    p.add_argument('--corner',choices=['TT','SS','FF'],default='TT',help='locked architecture Liberty corner; reports its actual operating point')
    p.add_argument('--constraints',type=Path,help='original complete controller SDC for an ECO checkpoint')
    run(p.parse_args())
