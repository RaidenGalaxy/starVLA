import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_sram_mode_timing import configuration_flops,mode_coverage


class StaticConfigurationTest(unittest.TestCase):
    def test_only_the_explicit_static_configuration_endpoints_are_excluded(self):
        records=[dict(pin='cfg'+str(i)+'/QN')for i in range(4)]
        raw='Warning: There are 4 unconstrained endpoints.\n'+''.join('  cfg'+str(i)+'/D\n'for i in range(4))
        self.assertTrue(mode_coverage(raw,records)['dynamic_endpoints_covered'])
        raw=raw.replace('cfg3/D','request/D')
        self.assertFalse(mode_coverage(raw,records)['dynamic_endpoints_covered'])

    def test_unrecognized_endpoints_or_warnings_do_not_disappear_from_coverage(self):
        self.assertFalse(mode_coverage('Warning: There is 1 unconstrained endpoint.\n  output_port\n',[])['dynamic_endpoints_covered'])
        self.assertFalse(mode_coverage('Warning: There is 1 register without a clock.\n',[])['dynamic_endpoints_covered'])

    def fixture(self):
        return '''
BUF x0(.A(sdel[0]),.Y(d0));
DFFHQN f0(.CLK(clk),.D(d0),.QN(q0));
INV x1(.A(sdel[1]),.Y(d1));
DFFHQN f1(.CLK(clk),.D(d1),.QN(q1));
HB x2(.A(sdel[2]),.Y(d2));
DFFHQ f2(.CLK(clk),.D(d2),.Q(q2));
INV x3(.A(sdel[3]),.Y(d3));
DFFHQ f3(.CLK(clk),.D(d3),.Q(q3));
DFFHQ dynamic(.CLK(clk),.D(ce_n),.Q(read_q));
'''

    def test_inversion_and_qn_polarity_preserve_the_hardwired_zero_mode(self):
        records=configuration_flops(self.fixture())
        self.assertEqual([r['case_value']for r in records],[1,0,0,1])
        self.assertEqual([r['pin']for r in records],['f0/QN','f1/QN','f2/Q','f3/Q'])

    def test_a_dynamic_logic_cone_is_not_treated_as_a_constant_mode_flop(self):
        raw=self.fixture().replace('BUF x0(.A(sdel[0]),.Y(d0));',
            'AND2 x0(.A(sdel[0]),.B(ce_n),.Y(d0));')
        with self.assertRaisesRegex(ValueError,'exactly four'):configuration_flops(raw)


if __name__=='__main__':unittest.main()
