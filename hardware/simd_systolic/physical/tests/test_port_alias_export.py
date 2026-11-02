import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'physical'))
from asap7_platform import profile
from run import sha,tool


class PortAliasExportTest(unittest.TestCase):
    def test_actual_shadowed_bus_bit_export_and_all_input_values(self):
        technology=profile('TT')
        with tempfile.TemporaryDirectory()as directory:
            out=Path(directory);net=out/'probe.v'
            net.write_text('module PortAliasProbe(input [2:0] d,output [2:0] q);\n'+
                ''.join(f'BUFx2_ASAP7_75t_R b{i}(.A(d[{i}]),.Y(q[{i}]));\n'for i in range(3))+'endmodule\n')
            script=out/'export.tcl';script.write_text(f'''read_lef {{{technology['tech_lef']}}}
read_lef {{{technology['cell_lef']}}}
read_liberty {{{technology['liberty']}}}
read_verilog {{{net}}}
link_design PortAliasProbe
initialize_floorplan -die_area {{0 0 20 20}} -core_area {{1 1 19 19}} -site asap7sc7p5t
set block [ord::get_db_block]
if {{![[$block findNet {{q[2]}}] rename separate_q2]}} {{error "fixture rename failed"}}
if {{![[$block findNet {{q[0]}}] rename {{q[2]}}]}} {{error "fixture shadow failed"}}
foreach inst [$block getInsts] {{$inst setDoNotTouch true}}
write_verilog {{{out/'raw.v'}}}
source {{{ROOT/'physical/canonical_net_names.tcl'}}}
source {{{ROOT/'physical/canonical_port_aliases.tcl'}}}
simd_export_port_aliases {{{out}}}
''')
            local=ROOT/'.deps/tools/openroad'
            env={**os.environ,'LD_LIBRARY_PATH':':'.join([str(local/'opt/or-tools/lib'),
                str(local/'usr/lib/x86_64-linux-gnu'),os.environ.get('LD_LIBRARY_PATH','')])}
            result=subprocess.run([tool('openroad'),'-exit',str(script)],env=env,
                capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('SIMD_PORT_SHADOWING_NETS_EXPORTED 1',result.stdout)
            self.assertEqual(sha(out/'before.def'),sha(out/'restored.def'))
            self.assertEqual(len({sha(out/n)for n in('before_connectivity.tsv','export_connectivity.tsv','restored_connectivity.tsv')}),1)
            cells=out/'cells.v'
            subprocess.run(['yosys','-Q','-T','-p',f'read_liberty -ignore_miss_func {technology["liberty"]}; write_verilog {cells}'],
                check=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT,timeout=60)
            raw=subprocess.run(['verilator','--lint-only','-Wno-fatal','--top-module','PortAliasProbe',str(cells),str(out/'raw.v')],
                capture_output=True,text=True,timeout=60)
            self.assertEqual(raw.returncode,0,raw.stderr)
            self.assertIn('MULTIDRIVEN',raw.stderr)
            bench=out/'main.cpp';bench.write_text('''#include "VPortAliasProbe.h"
int main(){VPortAliasProbe d;for(int value=0;value<8;value++){d.d=value;d.eval();if(d.q!=value)return 1;}return 0;}
''')
            with(out/'build.log').open('w')as stream:
                subprocess.run(['verilator','--cc','--exe','--build','-j','4','-Wno-fatal',
                    '--top-module','PortAliasProbe','--Mdir',str(out/'obj'),str(cells),str(out/'repaired.v'),
                    str(bench),'-o',str(out/'sim')],check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
            self.assertNotIn('MULTIDRIVEN',(out/'build.log').read_text())
            subprocess.run([str(out/'sim')],check=True,timeout=30)
