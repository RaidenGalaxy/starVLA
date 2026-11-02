"""Exercise the production route/extraction boundary with actual OpenROAD."""
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'physical'))
from asap7_platform import profile
from run import tool
from parasitics import audit as audit_parasitics


class DetailedRouteCheckpointTest(unittest.TestCase):
    def test_actual_routed_geometry_resumes_extraction_without_routing_again(self):
        technology = profile('TT')
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            netlist = out / 'fixture.v'
            netlist.write_text('''module RouteCheckpointProbe(input core,memory,d,output q,other);
wire a,b;
DFFHQNx1_ASAP7_75t_R c0(.CLK(core),.D(d),.QN(a));
DFFHQNx1_ASAP7_75t_R c1(.CLK(core),.D(a),.QN(q));
DFFHQNx1_ASAP7_75t_R m0(.CLK(memory),.D(d),.QN(b));
DFFHQNx1_ASAP7_75t_R m1(.CLK(memory),.D(b),.QN(other));
endmodule
''')
            sdc = out / 'design.sdc'
            sdc.write_text('''create_clock -name core -period 1 [get_ports core]
create_clock -name memory -period 2 [get_ports memory]
set_input_delay .1 -clock core [get_ports d]
set_output_delay .1 -clock core [get_ports q]
set_output_delay .1 -clock memory [get_ports other]
set_input_transition .05 [get_ports d]
set_load 2 [get_ports {q other}]
''')
            local = ROOT / '.deps/tools/openroad'
            env = {**os.environ,
                'LD_LIBRARY_PATH': ':'.join([str(local / 'opt/or-tools/lib'),
                    str(local / 'usr/lib/x86_64-linux-gnu'), os.environ.get('LD_LIBRARY_PATH', '')]),
                'SIMD_OUT': str(out), 'SIMD_TECH_LEF': str(technology['tech_lef']),
                'SIMD_CELL_LEF': str(technology['cell_lef']), 'SIMD_LIB': str(technology['liberty']),
                'SIMD_NETLIST': str(netlist), 'SIMD_TOP': 'RouteCheckpointProbe', 'SIMD_SDC': str(sdc),
                'SIMD_SET_RC': str(technology['root'] / 'setRC.tcl'),
                'SIMD_TRACKS': str(technology['tracks']),
                'SIMD_RCX': str(technology['rcx']),
                'SIMD_PROTECTED_TIES': str(ROOT / 'physical/protected_ties.tcl'),
                'SIMD_PROTECTED_CLOCKS': str(ROOT / 'physical/protected_clocks.tcl'),
                'SIMD_DETAILED_ROUTE_ONLY': '1'}
            # Port buffering dominates the area of this four-flop fixture;
            # give it explicit space while running the production stage code.
            script = out / 'fixture.tcl'
            script.write_text('''rename initialize_floorplan original_initialize_floorplan
proc initialize_floorplan {args} {
 original_initialize_floorplan -die_area {0 0 20 20} -core_area {2 2 18 18} -site asap7sc7p5t
}
source {''' + str(ROOT / 'physical/run.tcl') + '}\n')
            command = [tool('openroad'), '-exit', str(script)]
            first = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertIn('SIMD_DETAILED_ROUTE_CHECKPOINT_WRITTEN', first.stdout)
            self.assertIn('Complete detail routing.', first.stdout)
            self.assertNotIn('SIMD_FINAL_SETUP', first.stdout)
            self.assertFalse((out / 'design.spef').exists())
            paths = [out / name for name in ('post_detailed_route.odb', 'post_detailed_route.v', 'design.sdc')]
            saved = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            env.pop('SIMD_DETAILED_ROUTE_ONLY')
            env['SIMD_RESUME_DETAILED_ROUTE'] = '1'
            second = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn('SIMD_RESUMED_DETAILED_ROUTE_CHECKPOINT', second.stdout)
            self.assertNotIn('Complete detail routing.', second.stdout)
            self.assertIn('SIMD_FINAL_SETUP', second.stdout)
            self.assertIn('SIMD_FINAL_HOLD', second.stdout)
            self.assertGreater((out / 'design.spef').stat().st_size, 100)
            self.assertTrue((out / 'routed.v').exists())
            annotation=audit_parasitics(second.stdout,out/'design.spef')
            self.assertTrue(annotation['clean'],annotation)
            self.assertEqual(saved, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
            # Both original clock periods must survive a fresh tool process.
            self.assertRegex(second.stdout, r'core\s+1\.00\s')
            self.assertRegex(second.stdout, r'memory\s+2\.00\s')
            # OpenSTA may exit successfully after a malformed SPEF. The
            # production audit must still invalidate that timing evidence.
            bad=out/'broken.spef'
            bad.write_text('*SPEF "ieee 1481-1999"\n*NAME_MAP\n*1 bad_name\\\n\n*D_NET *1 .1\n*END\n')
            broken=out/'broken.tcl'
            broken.write_text(f'''read_liberty {{{technology['liberty']}}}
read_db {{{out/'design.odb'}}}
set_cmd_units -time ns -capacitance fF
read_sdc {{{sdc}}}
puts SIMD_SPEF_READ_BEGIN
read_spef {{{bad}}}
report_parasitic_annotation -report_unannotated
puts SIMD_SPEF_READ_END
report_worst_slack -max
''')
            failed_read=subprocess.run([tool('openroad'),'-exit',str(broken)],env=env,
                capture_output=True,text=True,timeout=60)
            self.assertEqual(failed_read.returncode,0,failed_read.stdout+failed_read.stderr)
            self.assertIn('STA-1657',failed_read.stdout+failed_read.stderr)
            self.assertFalse(audit_parasitics(failed_read.stdout+failed_read.stderr,bad)['clean'])
