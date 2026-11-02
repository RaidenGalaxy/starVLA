import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'physical'))
from asap7_platform import profile
from run import tool


class ResetRepairTest(unittest.TestCase):
    def run_case(self,mixed,control=False):
        technology=profile('TT')
        with tempfile.TemporaryDirectory()as directory:
            out=Path(directory)
            text='module ResetProbe(input clk,rst,d,output [31:0] q);wire rn,one;\n'
            text+='INVx1_ASAP7_75t_R driver(.A(rst),.Y(rn));TIEHIx1_ASAP7_75t_R hi(.H(one));\n'
            text+='\n'.join(f'DFFASRHQNx1_ASAP7_75t_R f{i}(.CLK(clk),.D(rn),.QN(q[{i}]),.RESETN(one),.SETN(one));'if control else
                f'DFFHQNx1_ASAP7_75t_R f{i}(.CLK(clk),.D(rn),.QN(q[{i}]));'if mixed and i else
                f'DFFASRHQNx1_ASAP7_75t_R f{i}(.CLK(clk),.D(d),.QN(q[{i}]),.RESETN(rn),.SETN(one));'for i in range(32))
            (out/'probe.v').write_text(text+'\nendmodule\n')
            script=f'''read_lef {{{technology['tech_lef']}}}
read_lef {{{technology['cell_lef']}}}
read_liberty {{{technology['liberty']}}}
read_verilog {{{out/'probe.v'}}}
link_design ResetProbe
set_cmd_units -time ns -capacitance fF
initialize_floorplan -die_area {{0 0 30 30}} -core_area {{1 1 29 29}} -site asap7sc7p5t
set block [ord::get_db_block]
set dbu [$block getDbUnitsPerMicron]
set k 0
foreach inst [$block getInsts] {{
 $inst setOrient R0
 $inst setLocation [expr {{int((2+($k%8)*3)*$dbu)}}] [expr {{int((2+($k/8)*2)*$dbu)}}]
 $inst setPlacementStatus PLACED
 incr k
}}
set_dont_touch [get_cells f*]
create_clock -name core -period 1 [get_ports clk]
set_input_transition .05 [get_ports {{rst d}}]
source {{{technology['root']/'setRC.tcl'}}}
set_wire_rc -signal -layer M3
set_wire_rc -clock -layer M5
estimate_parasitics -placement
source {{{ROOT/'physical/reset_repair.tcl'}}}
source {{{ROOT/'physical/control_repair.tcl'}}}
set targets [{f'simd_repair_control_fanout {{rn clk one}} 16 2 20'if control else'simd_repair_reset_fanout 16 20'} {{{out/'nets.tsv'}}}]
if {{[llength $targets] != {1 if mixed or control else 32}}} {{error "wrong repair target count"}}
set changed 0
foreach inst [$block getInsts] {{
 if {{[string match BUF* [[$inst getMaster] getName]]}} {{incr changed}}
}}
if {{$changed == 0 && [[[$block findInst driver] getMaster] getName] eq "INVx1_ASAP7_75t_R"}} {{error "electrical repair made no actual change"}}
for {{set i 0}} {{$i<32}} {{incr i}} {{
 set inst [$block findInst f$i]
 if {{![$inst isDoNotTouch]}} {{error "reset sink protection lost"}}
 if {{[[[$inst findITerm CLK] getNet] getName] ne "clk"}} {{error "clock changed"}}
 if {{{1 if control else 0} || ({1 if mixed else 0} && $i>0)}} {{
  if {{[simd_reset_logic_root [[$inst findITerm D] getNet]] ne {{PORT:rst 1}}}} {{error "shared reset/data logic changed"}}
  if {{{1 if control else 0}}} {{
   foreach name {{RESETN SETN}} {{
    if {{[[[$inst findITerm $name] getNet] getName] ne "one"}} {{error "constant tie network was repaired as a control"}}
   }}
  }}
 }} else {{
  if {{[[[$inst findITerm D] getNet] getName] ne "d"}} {{error "data changed"}}
  if {{[simd_reset_logic_root [[$inst findITerm RESETN] getNet]] ne {{PORT:rst 1}}}} {{error "reset logic or polarity changed"}}
 }}
}}
rename rsz::repair_net_cmd rsz::original_repair_net_cmd
proc rsz::repair_net_cmd {{args}} {{error "intentional repair failure"}}
if {{![catch {{{f'simd_repair_control_fanout {{rn}} 1 2 20'if control else'simd_repair_reset_fanout 1 20'} {{{out/'failure.tsv'}}}}}]}} {{error "repair error swallowed"}}
for {{set i 0}} {{$i<32}} {{incr i}} {{
 if {{![[$block findInst f$i] isDoNotTouch]}} {{error "protection not restored on failure"}}
}}
# A new snapshot must discard the old logic-root cache. Deliberately change
# the common driver input inside the repair stub; the guard must reject it.
rename rsz::repair_net_cmd {{}}
proc rsz::repair_net_cmd {{args}} {{
 set block [ord::get_db_block]
 [[$block findInst driver] findITerm A] connect [$block findNet d]
}}
if {{![catch {{{f'simd_repair_control_fanout {{rn}} 1 2 20'if control else'simd_repair_reset_fanout 1 20'} {{{out/'changed.tsv'}}}}} reason]}} {{error "changed logic escaped the fresh-cache check"}}
if {{![string match *changed* $reason]}} {{error "unexpected guard failure: $reason"}}
for {{set i 0}} {{$i<32}} {{incr i}} {{
 if {{![[$block findInst f$i] isDoNotTouch]}} {{error "protection not restored after logic rejection"}}
}}
puts SIMD_RESET_REPAIR_REGRESSION_PASSED
'''
            (out/'probe.tcl').write_text(script)
            local=ROOT/'.deps/tools/openroad'
            env={**os.environ,'LD_LIBRARY_PATH':':'.join([str(local/'opt/or-tools/lib'),
                str(local/'usr/lib/x86_64-linux-gnu'),os.environ.get('LD_LIBRARY_PATH','')])}
            result=subprocess.run([tool('openroad'),'-exit',str(out/'probe.tcl')],
                env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('SIMD_RESET_REPAIR_REGRESSION_PASSED',result.stdout)

    def test_actual_per_net_repair_preserves_protected_state_and_restores_on_error(self):
        self.run_case(False)

    def test_one_reset_pin_and_many_other_loads_still_select_the_heavy_net(self):
        self.run_case(True)

    def test_normal_data_fanout_is_repaired_without_touching_clock_or_logic(self):
        self.run_case(False,control=True)
