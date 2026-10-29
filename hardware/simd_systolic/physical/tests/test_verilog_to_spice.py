import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verilog_to_spice import convert


class PinOrderTest(unittest.TestCase):
    def fixture(self,root,text):
        (root/'cdl').mkdir();(root/'lib').mkdir()
        cdl=root/'cdl/cells.cdl'
        cdl.write_text('.SUBCKT XBUF Z VSS A VDD\nM0 Z A VSS VSS nmos\n.ENDS\n')
        (root/'lib/cells.lib').write_text('library (test) { cell (XBUF) { pin (Z) { direction : output; } } }')
        net=root/'mapped.v';net.write_text(text)
        return net,cdl,root/'out.sp'

    def test_real_cdl_order_power_and_bus_names(self):
        with tempfile.TemporaryDirectory()as directory:
            paths=self.fixture(Path(directory),'module ctrl_decode(input [0:0] a,output [0:0] z); XBUF u(.A(a[0]),.Z(z[0])); endmodule')
            report=convert(*paths)
            self.assertIn('X0 z[0] VSS a[0] VDD XBUF',paths[-1].read_text())
            self.assertEqual(report['electrical_cells'],1)
            self.assertFalse(report['lvs_passed'])

    def test_missing_input_fails_closed(self):
        with tempfile.TemporaryDirectory()as directory:
            paths=self.fixture(Path(directory),'module ctrl_decode(output z); XBUF u(.Z(z)); endmodule')
            with self.assertRaisesRegex(ValueError,'missing/non-scalar pin'):convert(*paths)

    def test_floating_output_is_confirmed_by_liberty(self):
        with tempfile.TemporaryDirectory()as directory:
            paths=self.fixture(Path(directory),'module ctrl_decode(input a); XBUF u(.A(a),.Z()); endmodule')
            report=convert(*paths)
            self.assertEqual(report['intentionally_open_outputs'],['u/Z'])
            self.assertIn('open_0_Z',paths[-1].read_text())


if __name__=='__main__':unittest.main()
