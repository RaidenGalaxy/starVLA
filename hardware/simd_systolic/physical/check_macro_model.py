#!/usr/bin/env python3
"""Compare an exported timing model with every extracted boundary bit path.

This qualifies a sampled slew/load envelope, never the enclosing design.
"""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess

from run import ROOT,sha,tool
from export_macro import tcl
from simd_eval.certification import core_certificate


def wrapper(top,lef):
    pins=re.findall(r'^  PIN (\S+)\s+ DIRECTION (INPUT|OUTPUT)\s*;',lef,re.M)
    if not pins:raise ValueError('abstract has no scalar boundary pins')
    groups={}
    for pin,direction in pins:
        match=re.fullmatch(r'([A-Za-z_]\w*)(?:\[(\d+)\])?',pin)
        if not match:raise ValueError('unsupported pin name '+pin)
        name,index=match[1],match[2]
        previous,indices=groups.setdefault(name,(direction,[]))
        if direction!=previous:raise ValueError('mixed bus directions '+name)
        indices.append(None if index is None else int(index))
    declarations=[];connections=[]
    for name,(direction,indices)in sorted(groups.items()):
        if indices==[None]:width=''
        elif None not in indices and sorted(indices)==list(range(max(indices)+1)):
            width=f'[{max(indices)}:0] '
        else:raise ValueError('noncontiguous boundary bus '+name)
        declarations.append(direction.lower()+' '+width+name)
        # OpenSTA reconstructs Liberty bus-bit groups. Vector named ports are
        # required; escaped scalar .\a[0] connections do not bind these buses.
        connections.append('.'+name+'('+name+')')
    return 'module BoundaryModelProbe('+','.join(declarations)+');\n'+top+' unit('+','.join(connections)+');\nendmodule\n',len(pins)


def check(abstract,out,slews,loads,tolerance):
    abstract=abstract.resolve();out=out.resolve()
    if not slews or not loads or any(not math.isfinite(v)or v<=0 for v in(*slews,*loads)):
        raise ValueError('positive finite slew/load samples required')
    if not math.isfinite(tolerance)or not 0<tolerance<=.01:
        raise ValueError('accuracy tolerance must be in (0, 0.01] ns')
    exported=json.loads((abstract/'verification.json').read_text())
    if not exported.get('passed')or exported.get('scope')!='routed_block_abstract_export':
        raise ValueError('an authenticated routed abstract export is required')
    source=Path(exported['source_directory'])
    guard=(source/'.lock').open('a');fcntl.flock(guard,fcntl.LOCK_SH|fcntl.LOCK_NB)
    timing_path=source/'timing.json';numeric_path=source/'gate_verification_routed.json'
    if not numeric_path.exists():numeric_path=source/'gate_verification.json'
    timing=json.loads(timing_path.read_text());numeric=json.loads(numeric_path.read_text())
    certificate=core_certificate(timing,numeric)
    if not certificate['passed']:raise ValueError('source routed certificate is no longer valid')
    for name,digest in exported['artifacts_sha256'].items():
        if sha(abstract/name)!=digest:raise ValueError('export artifact changed: '+name)
    if sha(timing_path)!=exported['source_timing_report_sha256']or sha(numeric_path)!=exported['source_numeric_report_sha256']:
        raise ValueError('source timing/numerical evidence changed after export')
    if sha(source/'design.odb')!=exported['source_database_sha256']:
        raise ValueError('source physical database changed after export')
    top=exported['top']
    if not re.fullmatch(r'[A-Za-z_]\w*',top):raise ValueError('unsupported top cell')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if(out/'verification.json').exists():raise ValueError('choose a new accuracy-check directory')
    net,pins=wrapper(top,(abstract/'block.lef').read_text())
    wrapper_path=out/'model_probe.v';wrapper_path.write_text(net)
    pdk=ROOT/'.deps/asap7';lib=Path(timing.get('liberty_file',pdk/'asap7_RVT_TT.lib'))
    evidence={k:Path(v)for k,v in timing['evidence'].items()}
    local=ROOT/'.deps/tools/openroad';env=os.environ.copy()
    env['LD_LIBRARY_PATH']=str(local/'opt/or-tools/lib')+':'+str(local/'usr/lib/x86_64-linux-gnu')
    common=f'''read_lef {tcl(pdk/'lef/asap7_tech_1x_201209.lef')}
read_lef {tcl(pdk/'lef/asap7sc7p5t_28_R_1x_220121a.lef')}
read_liberty {tcl(lib)}
'''
    files={'wrapper':wrapper_path,'export_report':abstract/'verification.json'}
    for mode in('flat','model'):
        loading=(f'''read_db {tcl(source/'design.odb')}
read_spef {tcl(evidence['spef'])}
'''if mode=='flat'else f'''read_lef {tcl(abstract/'block.lef')}
read_liberty {tcl(abstract/'block.lib')}
read_verilog {tcl(wrapper_path)}
link_design BoundaryModelProbe
''')
        csv_path=out/(mode+'.csv');script=out/(mode+'.tcl');log=out/(mode+'.log')
        script.write_text('set_thread_count 2\n'+common+loading+f'''read_sdc {tcl(evidence['sdc'])}
set_propagated_clock [all_clocks]
set fd [open {tcl(csv_path)} w]
puts $fd "slew_ns,load_ff,direction,path_delay,port,slack_ns"
foreach slew {{{' '.join(map(str,slews))}}} {{
 set_input_transition $slew $data_inputs
 foreach load {{{' '.join(map(str,loads))}}} {{
  set_load $load [all_outputs]
  foreach direction {{input output}} {{
   if {{$direction == "input"}} {{set ports $data_inputs;set side from}} else {{set ports [all_outputs];set side to}}
   foreach port $ports {{
    set name [get_full_name $port]
    foreach sense {{min max}} {{
     foreach edge {{rise fall}} {{
     # Compare the boundary bit's edge. max_rise/min_rise filter the
     # endpoint, which is an internal D pin in the flat reference but the
     # input boundary pin in the model; intervening inversion changes it.
     set option -${{edge}}_${{side}}
     set delay ${{sense}}_${{edge}}
     set paths [find_timing_paths $option $port -path_delay $sense -group_count 1]
     set slack "NA"
     if {{[llength $paths]}} {{
      # get_property returns command units (ns here), already formatted.
      # sta::format_time expects internal seconds and must not wrap this.
      set slack [get_property [lindex $paths 0] slack]
     }}
     puts $fd "$slew,$load,$direction,$delay,$name,$slack"
     }}
    }}
   }}
  }}
 }}
}}
close $fd
puts "SIMD_BOUNDARY_MODEL_CHECK_DONE"
''')
        with log.open('w')as stream:
            subprocess.run([tool('openroad'),'-exit',str(script)],stdout=stream,stderr=subprocess.STDOUT,env=env,check=True,timeout=3600)
        text=log.read_text()
        if 'SIMD_BOUNDARY_MODEL_CHECK_DONE'not in text or re.search(r'\[(?:WARNING|ERROR) STA-',text):
            raise ValueError('boundary reader/path warning; inspect '+str(log))
        for name,path in [('script',script),('log',log),('samples',csv_path)]:files[mode+'_'+name]=path
    def read(path):
        with path.open()as stream:rows=list(csv.DictReader(stream))
        values={}
        for row in rows:
            key=tuple(row[k]for k in('slew_ns','load_ff','direction','path_delay','port'))
            if key in values:raise ValueError('duplicate boundary sample')
            values[key]=None if row['slack_ns']=='NA'else float(row['slack_ns'])
        return values
    flat,model=read(out/'flat.csv'),read(out/'model.csv')
    if flat.keys()!=model.keys()or not flat:raise ValueError('boundary sample coverage differs')
    expected=(pins-1)*4*len(slews)*len(loads) # clock is excluded from data paths
    if len(flat)!=expected:raise ValueError(f'incomplete bit/edge samples: {len(flat)} != {expected}')
    differences=[];missing=[];inactive=0
    for key,value in flat.items():
        actual=model[key]
        if value is None and actual is None:inactive+=1;continue
        if value is None or actual is None:missing.append(key);continue
        differences.append(dict(zip(('slew_ns','load_ff','direction','path_delay','port'),key),
            flat_slack_ns=value,model_slack_ns=actual,error_ns=actual-value))
    maximum=max((abs(r['error_ns'])for r in differences),default=math.inf)
    passed=not missing and maximum<=tolerance
    report=dict(passed=passed,scope='routed_boundary_timing_model_accuracy',
        source_directory=str(source),export_directory=str(abstract),top=top,
        rtl_sha256=timing['rtl_sha256'],netlist_sha256=timing['netlist_sha256'],
        source_certificate=certificate,bit_ports=pins,slew_samples_ns=slews,load_samples_ff=loads,
        sampled_slew_load_points=len(slews)*len(loads),bit_edge_samples=len(flat),
        input_edges_filtered_at_source=True,output_edges_filtered_at_endpoint=True,
        active_bit_edge_samples=len(differences),inactive_bit_edge_samples=inactive,
        missing_or_extra_paths=missing,maximum_absolute_error_ns=maximum,
        accuracy_tolerance_ns=tolerance,worst_comparisons=sorted(differences,key=lambda r:abs(r['error_ns']),reverse=True)[:20],
        sampled_boundary_model_qualified=passed,internal_reg_to_reg_paths_covered_by_source_certificate=True,
        timing_validated=False,parent_design_certified=False,hierarchical_signoff_ready=False,
        limitations=['Only the sampled input-slew/output-load envelope is compared; no enclosing implementation is certified.',
            'Parent placement/routing, clock integration, CDC and all-format numerical verification remain required.'],
        files={name:str(path)for name,path in files.items()},artifacts_sha256={name:sha(path)for name,path in files.items()},checker_sha256=sha(Path(__file__)))
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    if not passed:raise AssertionError('exported boundary model differs from extracted paths')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('abstract',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--slews-ns',type=float,nargs='+',default=[.01,.05,.1])
    parser.add_argument('--loads-ff',type=float,nargs='+',default=[.5,3.898,15.])
    parser.add_argument('--tolerance-ns',type=float,default=.001)
    args=parser.parse_args();check(args.abstract,args.out,args.slews_ns,args.loads_ff,args.tolerance_ns)
