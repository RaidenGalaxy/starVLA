import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts'),str(ROOT/'physical')]
from simd_eval.architecture import ARCH, CONFIG, read_architecture, require_f2, validate_dram
from simd_eval.constraints import core_policy, system_policy
from memory_inventory import inventory
from audit_sram import audit, gds_structures
from system_constraints import system_sdc


class MigrationContractTest(unittest.TestCase):
    def test_two_stages_and_all_three_clock_periods_are_enforced(self):
        require_f2(2)
        for depth in (1,8,12):
            with self.assertRaises(ValueError):require_f2(depth)
        validate_dram(ROOT/'configs/ddr4.ini')
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'architecture.properties'
            for old,new in [('stages=2','stages=12'),('memory_period_ps=2000','memory_period_ps=4000')]:
                path.write_text(CONFIG.read_text().replace(old,new))
                with self.assertRaises(ValueError):read_architecture(path)

    def test_hard_macro_capacity_cannot_be_replaced_by_declared_wires(self):
        with tempfile.TemporaryDirectory()as directory:
            net=Path(directory)/'synth.v'
            instances=[]
            for name,count in [('a',2),('b',4),('c',2),('aggregates',8)]:
                instances.extend(f'srambank_256x4x64_6t122 \\tile.{name}.word{i}.macro (.clk(clk));'for i in range(count))
            net.write_text('\n'.join(instances))
            report=inventory(net)
            self.assertTrue(report['all_nominal_bits_retained'])
            self.assertEqual(sum(report['retained_bytes'].values()),131072)
            self.assertFalse(report['physical_macro_certified'])
            net.write_text('\n'.join(instances[:-1]))
            self.assertFalse(inventory(net)['all_nominal_bits_retained'])

    def test_missing_macro_physical_views_and_reports_never_certify(self):
        with tempfile.TemporaryDirectory()as directory:
            report=audit(Path(directory))
            self.assertFalse(report['physical_certified'])
            self.assertGreaterEqual(len(report['gaps']),8)
            path=Path(directory)/'padding.gds'
            path.write_bytes(bytes(2048))
            self.assertEqual(gds_structures(path),([],[]))

    def test_dual_clock_policy_rejects_extra_clock_group_exception(self):
        with tempfile.TemporaryDirectory()as directory:
            net=Path(directory)/'synth.v';cells=[]
            for fifo in ('waves','reads','results','dones','auxRequests','auxResults','seeds','seedAcks'):
                for direction in ('wg','rg'):
                    for bit in range(4):
                        for stage in ('','1','2'):
                            name=f'tile.{fifo}.{direction}{stage}[{bit}]'
                            previous=f'tile.{fifo}.{direction}{"" if stage=="1" else "1"}[{bit}]' if stage else 'input'
                            inst=f'{fifo}_{direction}_{stage or "source"}_{bit}'
                            cells.append(f'DFFHQNx1_ASAP7_75t_R {inst} (.D({previous}),.Q({name}));')
            net.write_text('\n'.join(cells));text=system_sdc(net,ROOT/'physical/cdc_constraints.tcl')
            self.assertTrue(system_policy(text,text)['clean'])
            self.assertFalse(system_policy(text+'set_clock_groups -asynchronous\n',text)['clean'])
            self.assertFalse(system_policy(text,None)['clean'])
            net.write_text('\n'.join(cells[:-1]))
            with self.assertRaises(ValueError):system_sdc(net,ROOT/'physical/cdc_constraints.tcl')


if __name__=='__main__':unittest.main()
