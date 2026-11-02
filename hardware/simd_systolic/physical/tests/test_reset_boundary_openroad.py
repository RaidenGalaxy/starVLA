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


class ResetBoundaryTest(unittest.TestCase):
    def test_actual_reset_pair_truth_data_chain_clock_and_rejection(self):
        technology = profile('TT')
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            net = out / 'probe.v'
            net.write_text('''module ResetBoundaryProbe(input io_coreClock,io_memoryClock,io_coreReset,io_memoryReset,output value);
wire epoch,rn,one,zero,a,first,b;
OR2x2_ASAP7_75t_R epoch_or(.A(io_coreReset),.B(io_memoryReset),.Y(epoch));
AND2x2_ASAP7_75t_R wrong_and(.A(io_coreReset),.B(io_memoryReset),.Y());
INVx1_ASAP7_75t_R reset_inv(.A(epoch),.Y(rn));
TIEHIx1_ASAP7_75t_R hi(.H(one));
TIELOx1_ASAP7_75t_R lo(.L(zero));
DFFASRHQNx1_ASAP7_75t_R \\probe/f0 (.CLK(io_coreClock),.D(zero),.QN(a),.RESETN(one),.SETN(rn));
INVx1_ASAP7_75t_R restore_first(.A(a),.Y(first));
DFFASRHQNx1_ASAP7_75t_R \\probe/f1 (.CLK(io_coreClock),.D(first),.QN(b),.RESETN(one),.SETN(rn));
INVx1_ASAP7_75t_R restore_second(.A(b),.Y(value));
endmodule
''')
            script = out / 'probe.tcl'
            script.write_text(f'''read_lef {{{technology['tech_lef']}}}
read_lef {{{technology['cell_lef']}}}
read_liberty {{{technology['liberty']}}}
read_verilog {{{net}}}
link_design ResetBoundaryProbe
source {{{ROOT / 'physical/reset_repair.tcl'}}}
source {{{ROOT / 'physical/protected_clocks.tcl'}}}
source {{{ROOT / 'physical/reset_boundary_audit.tcl'}}}
set stream [open {{{out / 'pairs.tsv'}}} w]
lassign [simd_audit_reset_pair probe io_coreClock $stream] first_name second_name
set first_sta [string map {{\\\\ ""}} $first_name]
set second_sta [string map {{\\\\ ""}} $second_name]
if {{[llength [get_cells [list $first_sta]]]!=1}} {{error "Actual first-stage name does not resolve in STA"}}
if {{[llength [get_pins [list "$second_sta/D"]]]!=1}} {{error "Actual second-stage D does not resolve in STA"}}
set block [ord::get_db_block]
set second [$block findInst $second_name]
[$second findITerm RESETN] connect [$block findNet rn]
[$second findITerm SETN] connect [$block findNet one]
if {{![catch {{simd_audit_reset_pair probe io_coreClock $stream}}]}} {{error "Legacy inverted reset mapping accepted"}}
[$second findITerm RESETN] connect [$block findNet one]
[$second findITerm SETN] connect [$block findNet rn]
set d [$second findITerm D]
$d connect [$block findNet zero]
if {{![catch {{simd_audit_reset_pair probe io_coreClock $stream}}]}} {{error "Broken stage data chain accepted"}}
$d connect [$block findNet first]
set clk [$second findITerm CLK]
$clk connect [$block findNet io_memoryClock]
if {{![catch {{simd_audit_reset_pair probe io_coreClock $stream}}]}} {{error "Mixed clock domains accepted"}}
$clk connect [$block findNet io_coreClock]
''' + '''set inv [$block findInst reset_inv]
set wrong [$block findInst wrong_and]
set alternate [odb::dbNet_create $block wrong_epoch]
[$wrong findITerm Y] connect $alternate
[$inv findITerm A] connect $alternate
if {![catch {simd_audit_reset_pair probe io_coreClock $stream}]} {error "Wrong AND reset function accepted"}
[$inv findITerm A] connect [$block findNet epoch]
simd_audit_reset_pair probe io_coreClock $stream
close $stream
puts SIMD_RESET_BOUNDARY_FIXTURE_PASSED
''')
            local = ROOT / '.deps/tools/openroad'
            env = {**os.environ, 'LD_LIBRARY_PATH': ':'.join([str(local / 'opt/or-tools/lib'),
                str(local / 'usr/lib/x86_64-linux-gnu'), os.environ.get('LD_LIBRARY_PATH', '')])}
            result = subprocess.run([tool('openroad'), '-exit', str(script)], env=env,
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('SIMD_RESET_BOUNDARY_FIXTURE_PASSED', result.stdout)
