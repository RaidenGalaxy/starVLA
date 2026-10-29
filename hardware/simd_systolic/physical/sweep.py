#!/usr/bin/env python3
"""Re-run STA at explicit periods on one extracted routed implementation.

This is a frequency sweep of a fixed layout, not an estimate from 1/(T-WNS).
It never certifies formats or hierarchy absent from the measured netlist.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).parent))
from run import tool,sha


def sweep(directory,periods):
    directory=directory.resolve();env=os.environ.copy();local=ROOT/'.deps/tools/openroad'
    if not periods or any(not math.isfinite(p)or p<=0 for p in periods):raise ValueError('positive finite clock periods required')
    # Frequency is a property of this routed implementation. Authenticate the
    # netlist, extraction and constraints before reusing its physical checks.
    proof=json.loads((directory/'timing.json').read_text())
    for name in('netlist','sdc','spef','electrical','coverage','coverage_analysis','drc'):
        if name not in proof.get('evidence',{}):
            raise RuntimeError('incomplete routed evidence; run physical/finalize.py first: '+name)
        path=Path(proof['evidence'][name])
        if sha(path)!=proof['evidence_sha256'][name]:raise RuntimeError('changed routed evidence: '+name)
    reset_waived=proof['scope']!='fp64'and bool(re.search(r'set_false_path[^\n]*\breset\b',Path(proof['evidence']['sdc']).read_text()))
    constraint_clean=proof.get('constraint_policy_clean',True)and not reset_waived
    static_clean=proof['coverage_clean']and proof['route_drc_clean']and constraint_clean
    env['LD_LIBRARY_PATH']=str(local/'opt/or-tools/lib')+':'+str(local/'usr/lib/x86_64-linux-gnu')
    original=Path(proof['evidence']['sdc']).read_text();constraints=[]
    pattern=r'(?m)^(create_clock\s+-name\s+core\s+-period\s+)[\d.]+(\s+\[get_ports\s+clock\])'
    for index,period in enumerate(periods):
        changed,count=re.subn(pattern,lambda m:m[1]+format(period,'.9f')+m[2],original)
        if count!=1:raise ValueError('expected one explicit core clock definition in authenticated SDC')
        path=directory/f'sweep_{index}.sdc';path.write_text(changed);constraints.append(path)
    tcl=directory/'sweep.tcl'
    tcl.write_text(f'''read_liberty {ROOT/'.deps/asap7/asap7_RVT_TT.lib'}
read_db {directory/'design.odb'}
read_sdc {directory/'design.sdc'}
read_spef {directory/'design.spef'}
set_propagated_clock [all_clocks]
set index 0
foreach period {{{' '.join(map(str,periods))}}} {{
 # Reapply the complete IO delays, loads and uncertainty to this clock.
 # A clock replacement must not silently discard constraints during a sweep.
 set constraints [lindex {{{' '.join(str(p)for p in constraints)}}} $index]
 read_sdc $constraints
 set_propagated_clock [all_clocks]
 set setup [sta::format_time [sta::worst_slack_cmd max] 9]
 set hold [sta::format_time [sta::worst_slack_cmd min] 9]
 puts "SIMD_SWEEP $period $setup $hold"
 report_tns -digits 9
 report_check_types -max_slew -max_capacitance -max_fanout -min_period -min_pulse_width -violators -digits 9 > {directory}/sweep_${{index}}_electrical.rpt
 incr index
}}
''')
    proc=subprocess.run([tool('openroad'),'-exit',str(tcl)],check=True,text=True,capture_output=True,env=env)
    (directory/'sweep.log').write_text(proc.stdout+proc.stderr)
    rows=[]
    blocks=re.findall(r'SIMD_SWEEP ([^\n]+)\n(.*?)(?=SIMD_SWEEP |\Z)',proc.stdout,re.S)
    for index,(fields,body)in enumerate(blocks):
        p,s,h=map(float,fields.split())
        tns=float(re.search(r'\btns\s+([-+\d.]+)',body)[1])
        electrical=directory/f'sweep_{index}_electrical.rpt'
        clean='VIOLATED'not in electrical.read_text().upper()
        setup_hold=s>=0 and h>=0 and tns==0
        rows.append(dict(period_ns=p,setup_wns_ns=s,hold_wns_ns=h,tns_ns=tns,
            setup_hold_closed=setup_hold,electrical_clean=clean,closed=setup_hold and clean and static_clean,
            sdc=str(constraints[index]),sdc_sha256=sha(constraints[index]),
            electrical_report=str(electrical),electrical_sha256=sha(electrical)))
    if len(rows)!=len(periods):raise RuntimeError('incomplete STA frequency sweep')
    valid=[r for r in rows if r['closed']]
    result={'layout':str(directory),'corners':['ASAP7 RVT TT'],'sweep':rows,
      'scope':proof['scope'],'stages':proof['stages'],'rows':proof['rows'],'cols':proof['cols'],
      'rtl_sha256':proof['rtl_sha256'],'netlist_sha256':sha(directory/'routed.v'),
      'spef_sha256':sha(directory/'design.spef'),'sdc_sha256':sha(directory/'design.sdc'),
      'timing_proof_sha256':sha(directory/'timing.json'),'script_sha256':sha(tcl),
      'route_drc_clean':proof['route_drc_clean'],'coverage_clean':proof['coverage_clean'],
      'constraint_policy_clean':constraint_clean,
      'full_constraints_reapplied_each_period':True,
      'maximum_tested_closed_frequency_hz':1e9/min(r['period_ns']for r in valid)if valid else None}
    (directory/'frequency_sweep.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path)
    p.add_argument('--periods',nargs='+',type=float,default=[1,1.5,2,2.5,3,4,5,6,8,10])
    a=p.parse_args();sweep(a.directory,a.periods)
