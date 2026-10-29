#!/usr/bin/env python3
"""Export LEF/Liberty from a core with authenticated extracted timing.

This exports an integration input, not a certificate for its parent design.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from run import ROOT,sha,tool
sys.path.insert(0,str(ROOT/'python'))
from simd_eval.certification import core_certificate


def tcl(value):
    value=str(value)
    for before,after in(('\\','\\\\'),('"','\\"'),('$','\\$'),('[','\\['),(']','\\]')):
        value=value.replace(before,after)
    return '"'+value+'"'


def export(source,out):
    source=source.resolve();out=out.resolve()
    if out==source:raise ValueError('export into a separate directory')
    guard=(source/'.lock').open('a');fcntl.flock(guard,fcntl.LOCK_SH|fcntl.LOCK_NB)
    timing_path=source/'timing.json'
    timing=json.loads(timing_path.read_text())
    numeric_path=source/'gate_verification_routed.json'
    if not numeric_path.exists():numeric_path=source/'gate_verification.json'
    numeric=json.loads(numeric_path.read_text())if numeric_path.exists()else{}
    certificate=core_certificate(timing,numeric)
    if not certificate['passed']:
        raise ValueError('source lacks a closed 1 ns routed certificate: '+', '.join(certificate['failures']))
    top=timing['top']
    if not re.fullmatch(r'[A-Za-z_]\w*',top):raise ValueError('unsupported physical block name')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'verification.json').unlink(missing_ok=True)
    odb=source/'design.odb'
    evidence={name:Path(path)for name,path in timing['evidence'].items()}
    pdk=ROOT/'.deps/asap7'
    lib=Path(timing.get('liberty_file',pdk/'asap7_RVT_TT.lib'))
    if sha(lib)!=timing['liberty_sha256']:raise ValueError('source process library changed')
    for original,name in((timing_path,'source_timing.json'),(numeric_path,'source_numeric.json')):
        shutil.copy2(original,out/name)
    script=out/'export.tcl'
    script.write_text(f'''set_thread_count 2
read_lef {tcl(pdk/'lef/asap7_tech_1x_201209.lef')}
read_lef {tcl(pdk/'lef/asap7sc7p5t_28_R_1x_220121a.lef')}
read_liberty {tcl(lib)}
read_db {tcl(odb)}
read_sdc {tcl(evidence['sdc'])}
read_spef {tcl(evidence['spef'])}
set_propagated_clock [all_clocks]
puts "SIMD_EXPORT_SOURCE_SETUP"
report_worst_slack -max -digits 9
puts "SIMD_EXPORT_SOURCE_HOLD"
report_worst_slack -min -digits 9
write_verilog {tcl(out/'odb_netlist.v')}
write_abstract_lef -bloat_occupied_layers {tcl(out/'block.lef')}
write_timing_model -library_name SimdQualifiedBlocks -cell_name {top} {tcl(out/'block.lib')}
exit
''')
    env=os.environ.copy();local=ROOT/'.deps/tools/openroad'
    env['LD_LIBRARY_PATH']=str(local/'opt/or-tools/lib')+':'+str(local/'usr/lib/x86_64-linux-gnu')
    def run(path,log):
        with log.open('w')as stream:
            subprocess.run([tool('openroad'),'-exit',str(path)],env=env,stdout=stream,
                stderr=subprocess.STDOUT,check=True,timeout=1800)
    run(script,out/'export.log')
    text=(out/'export.log').read_text()
    def slack(marker):
        match=re.search(marker+r'.*?worst slack\s+([-+\d.]+)',text,re.S)
        if not match:raise RuntimeError('missing source timing reproduction '+marker)
        return float(match[1])
    setup=slack('SIMD_EXPORT_SOURCE_SETUP');hold=slack('SIMD_EXPORT_SOURCE_HOLD')
    if abs(setup-timing['setup_wns_ns'])>1e-9 or abs(hold-timing['hold_wns_ns'])>1e-9:
        raise ValueError('reloaded physical database/SPEF does not reproduce source timing')
    if sha(out/'odb_netlist.v')!=timing['netlist_sha256']:
        raise ValueError('physical database connectivity differs from the verified routed netlist')
    lef=(out/'block.lef').read_text();model=(out/'block.lib').read_text()
    lef_pins=set(re.findall(r'^  PIN (\S+)',lef,re.M))
    lib_pins=set(re.findall(r'\bpin\s*\(\s*"?([^"\s)]+)"?\s*\)',model))
    if not lef_pins or lef_pins!=lib_pins:raise ValueError('LEF/Liberty bit-port coverage differs')
    if not all(name in model for name in('min_clock_tree_path','max_clock_tree_path')):
        raise ValueError('timing model does not preserve internal clock latency bounds')
    reload=out/'reload.tcl'
    reload.write_text(f'''read_lef {tcl(pdk/'lef/asap7_tech_1x_201209.lef')}
read_lef {tcl(out/'block.lef')}
read_liberty {tcl(out/'block.lib')}
if {{[llength [get_lib_cells */{top}]]!=1}} {{error "abstract cell missing"}}
puts "SIMD_ABSTRACT_RELOAD_OK"
exit
''')
    run(reload,out/'reload.log')
    if 'SIMD_ABSTRACT_RELOAD_OK'not in(out/'reload.log').read_text():raise ValueError('abstract reload failed')
    # Re-authenticate after export so mutable source files cannot be silently
    # combined with an earlier numerical or timing certificate.
    if not core_certificate(json.loads(timing_path.read_text()),json.loads(numeric_path.read_text()))['passed']:
        raise ValueError('source certificate changed during export')
    report=dict(passed=True,scope='routed_block_abstract_export',functional_scope=timing['scope'],
        top=top,stages=timing['stages'],rows=timing.get('rows'),cols=timing.get('cols'),
        source_directory=str(source),source_certificate=certificate,
        source_timing_validated=True,minimum_verified_clock_period_ns=timing['period_ns'],
        source_setup_wns_ns=setup,source_hold_wns_ns=hold,
        bit_ports_checked=len(lef_pins),internal_clock_latency_bounds_exported=True,
        database_matches_verified_routed_connectivity=True,abstract_reader_checked=True,
        rtl_sha256=timing['rtl_sha256'],netlist_sha256=timing['netlist_sha256'],
        liberty_sha256=sha(lib),source_database_sha256=sha(odb),source_spef_sha256=sha(evidence['spef']),
        source_timing_report_sha256=sha(timing_path),source_numeric_report_sha256=sha(numeric_path),
        files={name:str(out/name)for name in('block.lef','block.lib','odb_netlist.v')},
        artifacts_sha256={name:sha(out/name)for name in('block.lef','block.lib','odb_netlist.v','export.tcl','export.log','reload.tcl','reload.log')},
        exporter_sha256=sha(Path(__file__)),timing_validated=False,parent_design_certified=False,
        hierarchical_signoff_ready=False,
        limitations=['Parent placement, routing, propagated clocks and timing still require validation.',
            'Boundary-model accuracy over integration slews/loads has not been qualified.',
            'This export does not extend the source clock, format, process or numerical scope.'])
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();export(args.source,args.out)
