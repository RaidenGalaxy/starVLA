#!/usr/bin/env python3
"""Prove the balanced lane counter, then map a small timing diagnostic.

The isolated combinational probe is deliberately not a PE/array timing proof.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'physical'))
from run import mapping_library,tool


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check(out,baseline=None,mapped=False):
    out.mkdir(parents=True,exist_ok=True)
    primitive=out/'PopulationCount.sv'
    shutil.copy2(ROOT/'rtl/PopulationCount.sv',primitive)
    ports='input [15:0] bits,input [3:0] fmt,input enable,output [4:0] count'
    common='''wire [4:0] c16;wire [3:0] c8;wire [2:0] c4;wire [1:0] c2;
 wire [4:0] selected;
 assign selected=({5{fmt==0 || fmt==5}} & c16)
  |({5{fmt==1 || fmt==6}} & {1'b0,c8})
  |({5{fmt==2 || fmt==7 || fmt==8}} & {2'b0,c4})
  |({5{fmt==3 || fmt==9}} & {3'b0,c2})
  |({5{fmt==4 || fmt==10}} & {4'b0,bits[0]});
 assign count=enable?selected:5'b0;
'''
    actual=''.join(f'SimdPopulationCount{n} p{n}(bits[{n-1}:0],c{n});\n'for n in(2,4,8,16))
    # Independent behavioral reference: serial mathematical summation for
    # each prefix. Formal equivalence covers all masks, formats and enables.
    reference=''
    for n in(2,4,8,16):
        width=n.bit_length()
        reference+=f'assign c{n}='+' + '.join(f"{{{width-1}'b0,bits[{i}]}}"for i in range(n))+';\n'
    probes=out/'probes.sv'
    probes.write_text(f'module LaneCounterNew({ports});\n'+common+actual+'endmodule\n'+
        f'module LaneCounterReference({ports});\n'+common+reference+'endmodule\n'+'''
module PopulationMiter(input [15:0] bits,input [3:0] fmt,input enable,
 output equivalent);
 wire [4:0] actual,reference;
 wire [4:0] overflow,masked_before_count;
 reg [4:0] overflow_reference;
 integer i;
 LaneCounterNew dut(bits,fmt,enable,actual);
 LaneCounterReference ref_impl(bits,fmt,enable,reference);
 SimdPopulationCount16 ovf(bits,overflow);
 SimdPopulationCount16 old_gate(enable?bits:16'b0,masked_before_count);
 always @* begin
  overflow_reference=0;
  for(i=0;i<16;i=i+1)overflow_reference=overflow_reference+bits[i];
 end
 assign equivalent=(actual==reference) && (overflow==overflow_reference)
   && (masked_before_count==(enable?overflow:5'b0));
endmodule
''')
    proof=out/'proof.ys'
    proof.write_text(f'''read_verilog -sv {primitive} {probes}
hierarchy -check -top PopulationMiter
proc
setattr -mod -unset keep_hierarchy
setattr -unset keep_hierarchy
flatten
opt_clean
sat -set-def-inputs -prove equivalent 1 -verify
''')
    env=os.environ.copy();local=ROOT/'.deps/tools/openroad'
    env['LD_LIBRARY_PATH']=str(local/'opt/or-tools/lib')+':'+str(local/'usr/lib/x86_64-linux-gnu')
    def run(command,log,timeout=300):
        with log.open('w')as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,env=env,timeout=timeout)
    run(['yosys','-Q','-T','-s',str(proof)],out/'proof.log')
    if 'SAT proof finished - no model found: SUCCESS!'not in(out/'proof.log').read_text():
        raise AssertionError('missing successful population-count proof')
    report=dict(passed=True,scope='balanced_event_population_count',formal_all_inputs=True,
        mask_width=16,format_encodings_checked=16,legal_formats=11,enable_values_checked=2,
        mathematical_overflow_count_checked=True,unchanged_register_boundaries=True,
        overflow_valid_gate_after_count_equivalent=True,
        array_implementation_sha256=sha(ROOT/'src/main/scala/simd/Array.scala'),
        source_sha256=sha(primitive),proof_sha256=sha(proof),proof_log_sha256=sha(out/'proof.log'),
        checker_sha256=sha(Path(__file__)),timing_validated=False)
    def finish():
        (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
    if not mapped and baseline is None:
        finish();return
    identity=json.loads((baseline/'build_identity.json').read_text())if baseline else None
    lib=Path(identity['liberty_file'])if identity else ROOT/'.deps/asap7/asap7_RVT_TT.lib'
    if identity and sha(lib)!=identity['liberty_sha256']:raise RuntimeError('baseline process library changed')
    abc_lib,_=mapping_library(out,lib)
    mapping=out/'map.ys';net=out/'mapped.v'
    mapping.write_text(f'''read_liberty -lib {lib}
read_verilog -sv {primitive} {probes}
hierarchy -check -top LaneCounterNew
synth -top LaneCounterNew -flatten -noabc -noshare
abc -liberty {abc_lib} -D 850
clean
check -assert
write_verilog -noattr -noexpr {net}
setattr -mod -unset keep_hierarchy
flatten
write_json {out/'mapped.json'}
''')
    run(['yosys','-Q','-T','-s',str(mapping)],out/'map.log')
    sdc=out/'probe.sdc'
    sdc.write_text('''create_clock -name core -period 1.000
set_clock_uncertainty 0.050 [get_clocks core]
set_input_delay 0.100 -clock core [all_inputs]
set_output_delay 0.100 -clock core [all_outputs]
set_input_transition 0.050 [all_inputs]
set_load 3.898 [all_outputs]
''')
    sta=out/'timing.tcl'
    sta.write_text(f'''read_liberty {lib}
read_verilog {net}
link_design LaneCounterNew
read_sdc {sdc}
report_checks -path_delay max -group_count 5 -format full_clock_expanded -digits 9 > {out/'setup.rpt'}
report_worst_slack -max -digits 9
exit
''')
    run([tool('sta'),'-exit',str(sta)],out/'timing.log')
    # This measures the mapped parity structure, independently of timing.
    module=json.loads((out/'mapped.json').read_text())['modules']['LaneCounterNew']
    drivers={}
    for cell in module['cells'].values():
        inputs=[bit for port,bits in cell['connections'].items()if cell['port_directions'][port]=='input'for bit in bits]
        for port,bits in cell['connections'].items():
            if cell['port_directions'][port]=='output':
                for bit in bits:drivers[bit]=(cell['type'],inputs)
    cache={}
    def parity_depth(bit):
        if bit in cache:return cache[bit]
        if bit not in drivers:return 0
        kind,inputs=drivers[bit]
        if not kind.startswith(('XOR2_','XNOR2_')):return 0
        depth=1+max((parity_depth(i)for i in inputs),default=0)
        cache[bit]=depth;return depth
    depth=max((parity_depth(bit)for bit in drivers),default=0)
    lsb_depths={}
    for n in(2,4,8,16):
        outputs=module['netnames'][f'p{n}.count']['bits']
        levels=n.bit_length()-1
        lsb_depths[n]=parity_depth(outputs[0])
        if lsb_depths[n]!=levels:
            raise AssertionError(('mapped LSB parity tree collapsed',n,lsb_depths[n],levels))
        # A higher sum bit includes carry XORs as well as subtree parity.
        # Each extra adder boundary can contribute two XORs on that path;
        # this is a different bound from the log2(N) LSB parity depth.
        if max(map(parity_depth,outputs))>max(1,2*(levels-1)):
            raise AssertionError(('mapped carry/parity tree collapsed',n))
    slack=re.search(r'worst slack\s+([-+\d.]+)',(out/'timing.log').read_text())
    if not slack:raise AssertionError('missing isolated timing measurement')
    report.update(
        maximum_mapped_consecutive_xor_cells=depth,
        mapped_lsb_parity_depth_by_prefix=lsb_depths,
        isolated_combinational_setup_slack_ns=float(slack[1]),
        mapped_netlist_sha256=sha(net),liberty_sha256=sha(lib),sdc_sha256=sha(sdc),
        output_load_ff=3.898,setup_report_sha256=sha(out/'setup.rpt'),
        limitation='Isolated mapped combinational diagnostic only; PE/array extracted setup and hold remain required.')
    if baseline:
        measured=baseline/'pre_global_hold_setup.rpt'
        first_path=measured.read_text().split('\n\nStartpoint:')[0]
        report.update(baseline_array_netlist_sha256=identity['mapped_sha256'],
            baseline_array_rtl_sha256=identity['rtl_sha256'],baseline_report_sha256=sha(measured),
            baseline_report=str(measured),baseline_critical_path_xor_cells=len(re.findall(r'\((?:XOR|XNOR)2_',first_path)))
    finish()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=ROOT/'.build/population_count_proof')
    p.add_argument('--mapped',action='store_true',help='also run an isolated ASAP7 mapping/timing diagnostic')
    p.add_argument('--baseline',type=Path,help='optionally attach an actual array critical-path report; implies --mapped')
    a=p.parse_args();check(a.out.resolve(),a.baseline.resolve()if a.baseline else None,a.mapped)
