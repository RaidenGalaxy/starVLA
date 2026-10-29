import sys
import unittest
from pathlib import Path

import klayout.db as kdb

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from extend_sram_power import snapshot_region


class GeometrySnapshotTest(unittest.TestCase):
    def test_layout_mutation_cannot_change_the_baseline_region(self):
        layout=kdb.Layout();top=layout.create_cell('macro');layer=layout.layer(40,0)
        old=kdb.Box(0,0,100,20);new=kdb.Box(0,100,100,120)
        top.shapes(layer).insert(old)
        before=snapshot_region(top,layer)
        top.shapes(layer).insert(new)
        after=snapshot_region(top,layer)
        self.assertTrue((before^kdb.Region(old)).is_empty())
        self.assertTrue(((after-before)^kdb.Region(new)).is_empty())


if __name__=='__main__':unittest.main()
