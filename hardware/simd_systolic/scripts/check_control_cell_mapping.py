#!/usr/bin/env python3
"""Prove optional FF mappings using functional models from the actual Liberty."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def check(out):
    out.mkdir(parents=True,exist_ok=True)
    mapping=out/'NangateSyncControls.v';shutil.copy2(ROOT/'physical/NangateSyncControls.v',mapping)
    lib=ROOT/'.deps/nangate45/NangateOpenCellLibrary_typical.lib'
    cells=out/'cells.v';generate=out/'read_library.ys'
    generate.write_text(f'read_liberty -ignore_miss_func {lib}\nwrite_verilog {cells}\n')
    def run(script,log):
        with log.open('w')as stream:
            subprocess.run(['yosys','-Q','-T','-s',str(script)],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=120)
    run(generate,out/'read_library.log')
    miter=out/'miter.sv'
    miter.write_text('''module ControlCellMiter(input clock,data,enable,reset,output equivalent);
 reg enabled,reset_zero,reset_one;
 always @(posedge clock)begin
  if(enable)enabled<=data;
  if(reset)reset_zero<=0;else reset_zero<=data;
  if(reset)reset_one<=1;else reset_one<=data;
 end
 wire actual_enabled,actual_zero,actual_one;
 \\$_DFFE_PP_ hold_cell(.C(clock),.D(data),.E(enable),.Q(actual_enabled));
 \\$_SDFF_PP0_ zero_cell(.C(clock),.D(data),.R(reset),.Q(actual_zero));
 \\$_SDFF_PP1_ one_cell(.C(clock),.D(data),.R(reset),.Q(actual_one));
 assign equivalent=(enabled==actual_enabled)&&(reset_zero==actual_zero)&&(reset_one==actual_one);
endmodule
''')
    proof=out/'proof.ys';proof.write_text(f'''read_verilog -sv {cells} {mapping} {miter}
hierarchy -check -top ControlCellMiter
proc
flatten
opt_clean
sat -seq 6 -set-def-inputs -set-at 1 enable 1 -set-at 1 reset 1 -prove equivalent 1 -prove-skip 1 -verify
''')
    run(proof,out/'proof.log')
    if 'SUCCESS!'not in(out/'proof.log').read_text():raise AssertionError('missing successful sequential proof')
    induct=out/'induct.ys';induct.write_text(proof.read_text().split('sat -seq',1)[0]+'''
sat -seq 2 -set-def-inputs -tempinduct -tempinduct-inductonly -prove equivalent 1 -verify
''')
    run(induct,out/'induct.log')
    if 'SUCCESS!'not in(out/'induct.log').read_text():raise AssertionError('missing inductive step proof')
    # Also exercise the actual techmap pass, not just module instantiation.
    probe=out/'probe.v';probe.write_text('''module ControlMapProbe(input clock,data,enable,reset,output reg [2:0] q);
 always @(posedge clock)begin
  if(enable)q[0]<=data;
  if(reset)q[1]<=0;else q[1]<=data;
  if(reset)q[2]<=1;else q[2]<=data;
 end
endmodule
''')
    mapped=out/'mapped.json';script=out/'map.ys';script.write_text(f'''read_liberty -lib {lib}
read_verilog {probe}
hierarchy -top ControlMapProbe
synth -top ControlMapProbe -noabc -noshare
techmap -map {mapping}
clean
check -assert
write_json {mapped}
''')
    run(script,out/'map.log')
    design=json.loads(mapped.read_text())['modules']['ControlMapProbe'];actual=list(design['cells'].values())
    if len(actual)!=3 or any(c['type']!='SDFF_X1'for c in actual):raise AssertionError('techmap did not produce three actual library flops')
    files={'mapping':mapping,'liberty':lib,'cell_models':cells,'miter':miter,'proof':proof,
        'proof_log':out/'proof.log','inductive_script':induct,'inductive_log':out/'induct.log',
        'map_script':script,'mapped_design':mapped,'checker':Path(__file__)}
    report=dict(passed=True,scope='nangate_synchronous_control_cell_mapping',
        fine_cell_types=['$_DFFE_PP_','$_SDFF_PP0_','$_SDFF_PP1_'],physical_cell='SDFF_X1',
        clock_polarity_preserved=True,enable_and_synchronous_reset_preserved=True,
        arbitrary_initial_state=True,first_clock_forces_common_reference_state=True,
        sequential_induction_proven=True,actual_techmap_executed=True,unchanged_latency=True,
        timing_validated=False,
        limitation='Functional mapping proof only; all-format mapped numerical validation and extracted timing remain required.',
        files={k:str(p.resolve())for k,p in files.items()},
        artifacts_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest()for k,p in files.items()})
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    raise SystemExit('Nangate45 scan-control mapping is retired; ASAP7 uses standard sequential mapping')
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'artifacts/synchronous_control_cell_mapping')
    check(parser.parse_args().out.resolve())
