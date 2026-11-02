import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from netlist_fanout import census


class NetlistFanoutTest(unittest.TestCase):
    def test_first_pin_inline_and_multiline_bindings_and_clock_exclusion(self):
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'tool_export.v'
            path.write_text('''// .A(comment_is_not_a_net)
module Probe(input clock,a,output q);
BUFx2_ASAP7_75t_R b0 (.A(a),.Y(n0));
BUFx2_ASAP7_75t_R b1 (.A(a),
 .Y(n1));
DFFHQNx1_ASAP7_75t_R f0 (.CLK(clock),.D(n0),.QN(q));
DFFHQNx1_ASAP7_75t_R f1 (.CLK(clock),
 .D(\\escaped.net[3] ),.QN(n1));
INVx1_ASAP7_75t_R i0 (.A(port_bus[0]),.Y(\\escaped.net[3] )); // .A(comment_is_not_a_net)
INVx1_ASAP7_75t_R i1 (.A(\\tile._waves_rdata [71]),.Y(n2));
INVx1_ASAP7_75t_R i2 (.A(\\tile._waves_rdata [ 71 ] ),.Y(n3));
INVx1_ASAP7_75t_R i3 (.A(\\tile._waves_rdata[71] ),.Y(n4));
endmodule
''')
            refs,clocks=census(path)
            self.assertEqual(refs['a'],2)  # both bindings are on cell declaration lines
            self.assertEqual(refs['clock'],2)
            self.assertEqual(clocks,{'clock'})
            self.assertEqual(refs['escaped.net[3]'],2)
            self.assertEqual(refs['port_bus[0]'],1)
            self.assertEqual(refs['tile._waves_rdata[71]'],3)
            self.assertNotIn('comment_is_not_a_net',refs)
