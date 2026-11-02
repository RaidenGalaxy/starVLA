import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'physical'))
from asap7_platform import profile
from run import tool


class OpenStaGrayConstraintTest(unittest.TestCase):
    def test_cts_preserves_protected_clock_roots_and_data(self):
        technology=profile('TT')
        with tempfile.TemporaryDirectory()as directory:
            out=Path(directory)
            (out/'dut.v').write_text('''module ProtectedClockProbe(input core,memory,d,output q,other);
wire launch,first;
DFFHQNx1_ASAP7_75t_R launch_ff(.CLK(core),.D(d),.QN(launch));
DFFHQNx1_ASAP7_75t_R other_ff(.CLK(core),.D(launch),.QN(other));
DFFHQNx1_ASAP7_75t_R first_ff(.CLK(memory),.D(launch),.QN(first));
DFFHQNx1_ASAP7_75t_R second_ff(.CLK(memory),.D(first),.QN(q));
endmodule
''')
            script=f'''read_lef {{{technology['tech_lef']}}}
read_lef {{{technology['cell_lef']}}}
read_liberty {{{technology['liberty']}}}
read_verilog {{{out/'dut.v'}}}
link_design ProtectedClockProbe
set_cmd_units -time ns -capacitance fF
initialize_floorplan -die_area {{0 0 20 20}} -core_area {{1 1 19 19}} -site asap7sc7p5t
set block [ord::get_db_block]
set x 2000
foreach name {{launch_ff other_ff first_ff second_ff}} {{
 set inst [$block findInst $name]
 $inst setOrient R0
 $inst setLocation $x 2000
 $inst setPlacementStatus PLACED
 incr x 3000
}}
create_clock -name core -period 1 [get_ports core]
create_clock -name memory -period 2 [get_ports memory]
set_dont_touch [get_cells {{first_ff second_ff}}]
source {{{technology['root']/'setRC.tcl'}}}
set_wire_rc -signal -layer M3
set_wire_rc -clock -layer M5
source {{{ROOT/'physical/protected_clocks.tcl'}}}
simd_protected_cts -root_buf BUFx8_ASAP7_75t_R -buf_list {{BUFx2_ASAP7_75t_R BUFx4_ASAP7_75t_R BUFx8_ASAP7_75t_R}}
set count 0
foreach inst [$block getInsts] {{
 if {{[string match BUF* [[$inst getMaster] getName]]}} {{incr count}}
}}
if {{$count < 2}} {{error "fixture did not create both actual clock trees"}}
foreach name {{first_ff second_ff}} {{
 set inst [$block findInst $name]
 if {{![$inst isDoNotTouch]}} {{error "synchronizer protection lost"}}
 if {{[simd_clock_root [[$inst findITerm CLK] getNet]] ne {{memory 0}}}} {{error "protected clock changed domain or polarity"}}
}}
rename clock_tree_synthesis original_clock_tree_synthesis
proc clock_tree_synthesis {{args}} {{error "intentional CTS failure"}}
if {{![catch {{simd_protected_cts}}]}} {{error "CTS failure was swallowed"}}
foreach name {{first_ff second_ff}} {{
 if {{![[$block findInst $name] isDoNotTouch]}} {{error "protection not restored on CTS failure"}}
}}
puts SIMD_PROTECTED_CLOCK_REGRESSION_PASSED
'''
            (out/'probe.tcl').write_text(script)
            local=ROOT/'.deps/tools/openroad'
            env={**os.environ,'LD_LIBRARY_PATH':':'.join([str(local/'opt/or-tools/lib'),
                str(local/'usr/lib/x86_64-linux-gnu'),os.environ.get('LD_LIBRARY_PATH','')])}
            result=subprocess.run([tool('openroad'),'-exit',str(out/'probe.tcl')],
                env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('SIMD_PROTECTED_CLOCK_REGRESSION_PASSED',result.stdout)

    def test_tie_replication_preserves_protected_signals_and_inactive_controls(self):
        technology=profile('TT')
        with tempfile.TemporaryDirectory()as directory:
            out=Path(directory)
            (out/'dut.v').write_text('''module ProtectedTieProbe(input clk,d,output q);
wire one,first;
TIEHIx1_ASAP7_75t_R tie_one(.H(one));
DFFASRHQNx1_ASAP7_75t_R first_ff(.CLK(clk),.D(d),.QN(first),.RESETN(one),.SETN(one));
DFFASRHQNx1_ASAP7_75t_R second_ff(.CLK(clk),.D(first),.QN(q),.RESETN(one),.SETN(one));
endmodule
''')
            script=f'''read_lef {{{technology['tech_lef']}}}
read_lef {{{technology['cell_lef']}}}
read_liberty {{{technology['liberty']}}}
read_verilog {{{out/'dut.v'}}}
link_design ProtectedTieProbe
initialize_floorplan -utilization 30 -aspect_ratio 1 -core_space 3 -site asap7sc7p5t
set_dont_touch [get_cells {{first_ff second_ff}}]
source {{{ROOT/'physical/protected_ties.tcl'}}}
simd_repair_protected_ties
set block [ord::get_db_block]
foreach name {{first_ff second_ff}} {{
 set inst [$block findInst $name]
 if {{![$inst isDoNotTouch]}} {{error "synchronizer protection lost"}}
 foreach pin {{RESETN SETN}} {{
  set driver_count 0
  foreach term [[[$inst findITerm $pin] getNet] getITerms] {{
   if {{[[$term getMTerm] getIoType] ne "OUTPUT"}} {{continue}}
   if {{[[[$term getInst] getMaster] getName] ne "TIEHIx1_ASAP7_75t_R"}} {{error "constant changed"}}
   incr driver_count
  }}
  if {{$driver_count != 1}} {{error "constant no longer has one tie driver"}}
 }}
}}
if {{[[[$block findInst first_ff] findITerm D] getNet] ne [$block findNet d]}} {{error "first data net changed"}}
if {{[[[$block findInst second_ff] findITerm D] getNet] ne [$block findNet first]}} {{error "second data net changed"}}
rename repair_tie_fanout original_repair_tie_fanout
proc repair_tie_fanout {{args}} {{error "intentional tool failure"}}
if {{![catch {{simd_repair_protected_ties}}]}} {{error "tool failure was swallowed"}}
foreach name {{first_ff second_ff}} {{
 if {{![[$block findInst $name] isDoNotTouch]}} {{error "protection not restored on failure"}}
}}
puts SIMD_PROTECTED_TIE_REGRESSION_PASSED
'''
            (out/'probe.tcl').write_text(script)
            local=ROOT/'.deps/tools/openroad'
            env={**os.environ,'LD_LIBRARY_PATH':':'.join([str(local/'opt/or-tools/lib'),
                str(local/'usr/lib/x86_64-linux-gnu'),os.environ.get('LD_LIBRARY_PATH','')])}
            result=subprocess.run([tool('openroad'),'-exit',str(out/'probe.tcl')],
                env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('SIMD_PROTECTED_TIE_REGRESSION_PASSED',result.stdout)

    def test_real_launch_paths_are_bounded_and_second_stage_remains_timed(self):
        technology=profile('TT')
        with tempfile.TemporaryDirectory()as directory:
            out=Path(directory)
            (out/'dut.v').write_text('''module CdcStaProbe(input core,memory,d,output q);
wire launch,first;
DFFHQNx1_ASAP7_75t_R launch_ff(.CLK(core),.D(d),.QN(launch));
DFFHQNx1_ASAP7_75t_R first_ff(.CLK(memory),.D(launch),.QN(first));
DFFHQNx1_ASAP7_75t_R second_ff(.CLK(memory),.D(first),.QN(q));
endmodule
''')
            script=f'''read_lef {{{technology['tech_lef']}}}
read_lef {{{technology['cell_lef']}}}
read_liberty {{{technology['liberty']}}}
read_verilog {{{out/'dut.v'}}}
link_design CdcStaProbe
set_cmd_units -time ns -capacitance fF
create_clock -name core -period 1 [get_ports core]
create_clock -name memory -period 2 [get_ports memory]
# Deliberately different clock latencies expose an ineffective exception.
set_clock_latency 10 [get_clocks core]
set_clock_latency 5 [get_clocks memory]
source {{{ROOT/'physical/cdc_constraints.tcl'}}}
simd_constrain_gray [get_pins launch_ff/QN] [get_pins first_ff/D] [get_pins second_ff/D] 1
add_global_connection -net VDD -pin_pattern {{^VDD$}} -power
add_global_connection -net VSS -pin_pattern {{^VSS$}} -ground
global_connect
set block [ord::get_db_block]
set signal_before {{}}
foreach name {{first_ff second_ff}} {{
 set inst [$block findInst $name]
 foreach pin [$inst getITerms] {{
  set kind [[$pin getMTerm] getSigType]
  if {{$kind in {{POWER GROUND}}}} {{
   if {{[$pin getNet] ne "NULL"}} {{error "fixture did not reproduce skipped protected supplies"}}
  }} else {{lappend signal_before [$pin getNet]}}
 }}
}}
source {{{ROOT/'physical/protected_power.tcl'}}}
if {{[simd_connect_protected_supplies] != 4}} {{error "protected supplies not connected"}}
if {{[simd_connect_protected_supplies] != 0}} {{error "supply connection is not idempotent"}}
set signal_after {{}}
foreach name {{first_ff second_ff}} {{
 set inst [$block findInst $name]
 if {{![$inst isDoNotTouch]}} {{error "synchronizer protection lost"}}
 foreach pin [$inst getITerms] {{
  set kind [[$pin getMTerm] getSigType]
  if {{$kind eq "POWER"}} {{
   if {{[[$pin getNet] getName] ne "VDD"}} {{error "wrong protected supply"}}
   if {{![$pin isSpecial]}} {{error "protected supply is not a special pin"}}
  }} elseif {{$kind eq "GROUND"}} {{
   if {{[[$pin getNet] getName] ne "VSS"}} {{error "wrong protected ground"}}
   if {{![$pin isSpecial]}} {{error "protected ground is not a special pin"}}
  }} else {{lappend signal_after [$pin getNet]}}
 }}
}}
if {{$signal_before ne $signal_after}} {{error "signal connectivity changed"}}
puts "SIMD_PROTECTED_SUPPLIES_AND_SIGNALS_CHECKED"
'''
            for name,delay,launch,capture in [('cross_setup','max','launch','first'),
                    ('cross_hold','min','launch','first'),('sync_setup','max','first','second'),
                    ('sync_hold','min','first','second')]:
                script+=f'report_checks -path_delay {delay} -from [get_pins {launch}_ff/QN] -to [get_pins {capture}_ff/D] -digits 9 > {{{out/(name+".rpt")}}}\n'
            (out/'probe.tcl').write_text(script)
            local=ROOT/'.deps/tools/openroad'
            env={**os.environ,'LD_LIBRARY_PATH':':'.join([str(local/'opt/or-tools/lib'),
                str(local/'usr/lib/x86_64-linux-gnu'),os.environ.get('LD_LIBRARY_PATH','')])}
            result=subprocess.run([tool('openroad'),'-exit',str(out/'probe.tcl')],
                env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertNotIn('not a valid start point',result.stdout+result.stderr)
            self.assertIn('SIMD_PROTECTED_SUPPLIES_AND_SIGNALS_CHECKED',result.stdout)
            crossing=(out/'cross_setup.rpt').read_text()
            self.assertEqual(crossing.count('Startpoint:'),1)
            self.assertIn('Startpoint: launch_ff (',crossing)
            self.assertIn('max_delay',crossing)
            bound=float(re.search(r'([-\d.]+)\s+[-\d.]+\s+max_delay',crossing)[1])
            self.assertAlmostEqual(bound,.8,places=6)
            slack=float(re.search(r'([-\d.]+)\s+slack',crossing)[1])
            self.assertGreater(slack,.5)
            self.assertNotIn('Startpoint:',(out/'cross_hold.rpt').read_text())
            for name in('sync_setup','sync_hold'):
                report=(out/(name+'.rpt')).read_text()
                self.assertIn('Startpoint: first_ff (',report)
                self.assertIn('Path Group: memory',report)
                self.assertNotIn('max_delay',report)


if __name__=='__main__':unittest.main()
