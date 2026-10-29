import sys
import tempfile
import unittest
from pathlib import Path
import klayout.db as kdb

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_sram_controller import audit


class ControllerGeometryTest(unittest.TestCase):
    def fixture(self,root):
        reference=kdb.Layout();reference.dbu=0.001;layer=reference.layer(1,0)
        cell=reference.create_cell('BUF');cell.shapes(layer).insert(kdb.Box(0,0,20,40))
        library=root/'library.gds';reference.write(str(library))
        dut=kdb.Layout();dut.read(str(library));top=dut.create_cell('ctrl_decode')
        top.insert(kdb.CellInstArray(dut.cell('BUF').cell_index(),kdb.Trans(100,200)))
        gds=root/'dut.gds';dut.write(str(gds))
        net=root/'routed.v';net.write_text('module ctrl_decode(); BUF placed(); endmodule\n')
        return dut,gds,net,library

    def test_changed_cell_polygon_is_rejected(self):
        with tempfile.TemporaryDirectory()as directory:
            dut,gds,net,library=self.fixture(Path(directory))
            self.assertTrue(audit(gds,net,library)['passed'])
            dut.cell('BUF').shapes(dut.layer(1,0)).insert(kdb.Box(100,100,120,120));dut.write(str(gds))
            self.assertFalse(audit(gds,net,library)['passed'])

    def test_dropped_placed_instance_is_rejected(self):
        with tempfile.TemporaryDirectory()as directory:
            dut,gds,net,library=self.fixture(Path(directory))
            dut.cell('ctrl_decode').clear();dut.write(str(gds))
            self.assertFalse(audit(gds,net,library)['passed'])


if __name__=='__main__':unittest.main()
