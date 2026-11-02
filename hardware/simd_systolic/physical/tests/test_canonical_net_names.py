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


class CanonicalNetNamesTest(unittest.TestCase):
    def test_actual_db_rename_preserves_geometry_connections_ports_and_protection(self):
        technology=profile('TT')
        with tempfile.TemporaryDirectory()as directory:
            out=Path(directory)
            (out/'probe.v').write_text('''module CanonicalProbe(input clk,d,output q);
wire first;
DFFHQNx1_ASAP7_75t_R f0(.CLK(clk),.D(d),.QN(first));
DFFHQNx1_ASAP7_75t_R f1(.CLK(clk),.D(first),.QN(q));
endmodule
''')
            (out/'probe.tcl').write_text(rf'''read_lef {{{technology['tech_lef']}}}
read_lef {{{technology['cell_lef']}}}
read_liberty {{{technology['liberty']}}}
read_verilog {{{out/'probe.v'}}}
link_design CanonicalProbe
initialize_floorplan -die_area {{0 0 20 20}} -core_area {{2 2 18 18}} -site asap7sc7p5t
set block [ord::get_db_block]
set net [$block findNet first]
$net rename "prefix.tree\[12\]\nline"
$net setDoNotTouch true
set_dont_touch [get_cells f0]
source {{{ROOT/'physical/canonical_net_names.tcl'}}}
simd_export_canonical_net_names {{{out}}}
if {{![$net isDoNotTouch]}} {{error "Net protection changed"}}
if {{![[$block findInst f0] isDoNotTouch]}} {{error "Cell protection changed"}}
if {{[$net getName] ne "prefix.tree\[12\]\nline"}} {{error "Original net label not restored"}}
puts SIMD_CANONICAL_FIXTURE_PASSED
''')
            local=ROOT/'.deps/tools/openroad';env={**os.environ,'LD_LIBRARY_PATH':
                str(local/'opt/or-tools/lib')+':'+str(local/'usr/lib/x86_64-linux-gnu')}
            result=subprocess.run([tool('openroad'),'-exit',str(out/'probe.tcl')],env=env,
                capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('SIMD_CANONICAL_FIXTURE_PASSED',result.stdout)
            self.assertEqual((out/'before_rename.def').read_bytes(),(out/'restored_names.def').read_bytes())
            self.assertEqual((out/'before_connectivity.tcllist').read_bytes(),
                (out/'canonical_connectivity.tcllist').read_bytes())
            text=(out/'routed.v').read_text()
            self.assertIn('simd_export_internal_0',text)
            self.assertNotIn('prefix.tree',text)
            result=subprocess.run(['yosys','-Q','-T','-p',
                f'read_liberty -lib {technology["liberty"]}; read_verilog {out/"routed.v"}; hierarchy -check -top CanonicalProbe'],
                capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
