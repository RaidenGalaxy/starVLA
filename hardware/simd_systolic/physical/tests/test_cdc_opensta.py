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
