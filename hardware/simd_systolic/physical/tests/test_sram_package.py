import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'physical'),str(ROOT/'python')]
from sram_package import load_package,inventory
from simd_eval.architecture import ARCH


class PackageGuardTest(unittest.TestCase):
    def setUp(self):
        source=ROOT/'artifacts/migration_asap7_f2/sram_openfinram'
        if not(source/'package.json').is_file():self.skipTest('generate the research SRAM package first')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'package';shutil.copytree(source,self.root)

    def test_estimated_package_cannot_be_promoted_by_a_pass_flag(self):
        p=self.root/'package.json';data=json.loads(p.read_text());data['physical_certified']=True
        p.write_text(json.dumps(data))
        self.assertFalse(load_package(self.root,ARCH)['physical_certified'])

    def test_changed_nested_model_card_is_rejected(self):
        p=self.root/'models/7nm_TT.pm';p.write_text(p.read_text()+'\n* changed\n')
        with self.assertRaisesRegex(ValueError,'view changed'):load_package(self.root,ARCH)

    def test_changed_storage_period_is_rejected(self):
        p=self.root/'package.json';data=json.loads(p.read_text());data['configuration']['memory_period_ps']=2500
        p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'differs from the architecture'):load_package(self.root,ARCH)

    def test_missing_macro_is_visible_to_capacity_audit(self):
        native='sram_x256x64x1';cells=[]
        for bank,count in [('a',2),('b',4),('c',2),('aggregates',8)]:
            for i in range(count):cells.append(native+f' \\tile.{bank}.depthBanks[{i}].words[0].storage ();')
        net=Path(self.temp.name)/'mapped.v';net.write_text('\n'.join(cells))
        self.assertTrue(inventory(net,native)['all_nominal_bits_retained'])
        net.write_text('\n'.join(cells[:-1]))
        self.assertFalse(inventory(net,native)['all_nominal_bits_retained'])


if __name__=='__main__':unittest.main()
