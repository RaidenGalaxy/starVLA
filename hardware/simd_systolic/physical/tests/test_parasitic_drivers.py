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


class ParasiticDriversTest(unittest.TestCase):
    def test_actual_connected_drivers_cannot_be_treated_as_floating(self):
        t=profile('TT')
        with tempfile.TemporaryDirectory()as directory:
            p=Path(directory)
            (p/'probe.v').write_text('''module Probe(input d,unused,output q);
wire middle,dangling;
INVx1_ASAP7_75t_R a(.A(d),.Y(middle));
INVx1_ASAP7_75t_R b(.A(middle),.Y(q));
INVx1_ASAP7_75t_R c(.A(d),.Y(dangling));
INVx1_ASAP7_75t_R e(.A(d),.Y());
endmodule
''')
            (p/'probe.tcl').write_text(f'''read_lef {{{t['tech_lef']}}}
read_lef {{{t['cell_lef']}}}
read_liberty {{{t['liberty']}}}
read_verilog {{{p/'probe.v'}}}
link_design Probe
source {{{ROOT/'physical/parasitic_drivers.tcl'}}}
foreach name {{a/Y b/Y d}} {{
 lassign [simd_audit_parasitic_driver $name] kind sinks routed
 if {{$kind ne "connected_or_routed" || $sinks<1}} {{error "Connected signal incorrectly exempted: $name"}}
}}
foreach name {{c/Y e/Y unused}} {{
 lassign [simd_audit_parasitic_driver $name] kind sinks routed
 if {{$kind ni {{unconnected_output zero_fanout_no_routing}} || $sinks || $routed}} {{error "Zero-fanout signal classification failed: $name"}}
}}
if {{[lindex [simd_audit_parasitic_driver missing/Y] 0] ne "unknown_cell"}} {{error "Unknown driver accepted"}}
puts SIMD_PARASITIC_DRIVER_FIXTURE_PASSED
''')
            local=ROOT/'.deps/tools/openroad';env={**os.environ,'LD_LIBRARY_PATH':
                str(local/'opt/or-tools/lib')+':'+str(local/'usr/lib/x86_64-linux-gnu')}
            r=subprocess.run([tool('openroad'),'-exit',str(p/'probe.tcl')],env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            self.assertIn('SIMD_PARASITIC_DRIVER_FIXTURE_PASSED',r.stdout)
