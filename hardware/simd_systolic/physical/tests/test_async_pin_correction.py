import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'physical'))
from asap7_platform import profile
from correct_system_async import netlist_records
from run import sha,tool


class AsyncPinCorrectionTest(unittest.TestCase):
    def test_actual_odb_pin_correction_invariance_and_dual_control_rejection(self):
        technology=profile('TT')
        with tempfile.TemporaryDirectory()as directory:
            out=Path(directory);net=out/'probe.v'
            net.write_text('''module AsyncPinProbe(input clock,reset,data,output [7:0] qn);
wire rn,one;
INVx1_ASAP7_75t_R ri(.A(reset),.Y(rn));
TIEHIx1_ASAP7_75t_R hi(.H(one));
'''+''.join(f'DFFASRHQNx1_ASAP7_75t_R f{i}(.CLK(clock),.D(data),.QN(qn[{i}]),.RESETN(rn),.SETN(one));\n'for i in range(8))+'endmodule\n')
            script=out/'probe.tcl';script.write_text(f'''read_lef {{{technology['tech_lef']}}}
read_lef {{{technology['cell_lef']}}}
read_liberty {{{technology['liberty']}}}
read_verilog {{{net}}}
link_design AsyncPinProbe
source {{{ROOT/'physical/reset_repair.tcl'}}}
source {{{ROOT/'physical/correct_async_pins.tcl'}}}
set block [ord::get_db_block]
set flop [$block findInst f0]
[$flop findITerm SETN] connect [$block findNet reset]
if {{![catch {{simd_correct_async_pins 8 {{{out/'rejected.tsv'}}}}}]}} {{error "Independent active async inputs accepted"}}
if {{[[$flop findITerm RESETN] getNet] ne [$block findNet rn]}} {{error "Rejected correction mutated database"}}
[$flop findITerm SETN] connect [$block findNet one]
foreach inst [$block getInsts] {{
 if {{[[$inst getMaster] getName] eq "DFFASRHQNx1_ASAP7_75t_R"}} {{$inst setDoNotTouch true}}
}}
write_verilog {{{out/'before.v'}}}
simd_async_geometry {{{out/'before.tsv'}}}
simd_correct_async_pins 8 {{{out/'swaps.tsv'}}}
simd_async_geometry {{{out/'after.tsv'}}}
write_verilog {{{out/'after.v'}}}
foreach inst [$block getInsts] {{
 if {{[[$inst getMaster] getName] ne "DFFASRHQNx1_ASAP7_75t_R"}} {{continue}}
 if {{[[$inst findITerm RESETN] getNet] ne [$block findNet one] || [[$inst findITerm SETN] getNet] ne [$block findNet rn]}} {{error "Wrong corrected async pin"}}
}}
puts SIMD_ASYNC_PIN_FIXTURE_PASSED
''')
            local=ROOT/'.deps/tools/openroad'
            env={**os.environ,'LD_LIBRARY_PATH':':'.join([str(local/'opt/or-tools/lib'),
                str(local/'usr/lib/x86_64-linux-gnu'),os.environ.get('LD_LIBRARY_PATH','')])}
            result=subprocess.run([tool('openroad'),'-exit',str(script)],env=env,
                capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('SIMD_ASYNC_PIN_FIXTURE_PASSED',result.stdout)
            self.assertEqual(sha(out/'before.tsv'),sha(out/'after.tsv'))
            self.assertEqual(list(netlist_records(out/'before.v',True)),list(netlist_records(out/'after.v')))
