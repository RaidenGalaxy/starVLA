import sys
import tempfile
import subprocess
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mapped_system_constraints import crossings,system_sdc


class MappedSynchronizerTest(unittest.TestCase):
    def fixture(self):
        cells=[]
        for fifo in ('waves','reads','results','dones','auxRequests','auxResults','seeds','seedAcks'):
            for direction in ('wg','rg'):
                for bit in range(4):
                    prefix=f'tile.{fifo}';tag=f'{fifo}_{direction}_{bit}'
                    cells.extend([
                        f'DFFHQNx1_ASAP7_75t_R {tag}_launch (.D(input),.Q({tag}_q));',
                        f'INVx1_ASAP7_75t_R {tag}_out (.A({tag}_q),.Y({prefix}.{direction}[{bit}]));',
                        f'DFFHQNx1_ASAP7_75t_R {tag}_first (.D({prefix}.{direction}[{bit}]),.Q({tag}_firstq));',
                        f'INVx1_ASAP7_75t_R {tag}_restore (.A({tag}_firstq),.Y({prefix}.{direction}1[{bit}]));',
                        f'DFFHQNx1_ASAP7_75t_R {tag}_second (.D({prefix}.{direction}1[{bit}]),.Q({tag}_secondq));',
                        f'BUFx2_ASAP7_75t_R {tag}_buffer (.A({tag}_secondq),.Y({prefix}.{direction}2[{bit}]));'])
        return cells

    def test_launch_and_both_stages_resolve_through_polarity_cells(self):
        with tempfile.TemporaryDirectory()as directory:
            net=Path(directory)/'mapped.v';net.write_text('\n'.join(self.fixture()))
            records=crossings(net);self.assertEqual(len(records),64)
            self.assertTrue(all(r['launch'].endswith('_launch/Q')for r in records))
            self.assertTrue(all(r['first'].endswith('_first/D')for r in records))
            self.assertTrue(all(r['second'].endswith('_second/D')for r in records))
            self.assertEqual(system_sdc(net,'helper.tcl').count('\nsimd_constrain_gray '),16)

    def test_missing_synchronizer_stage_fails_closed(self):
        with tempfile.TemporaryDirectory()as directory:
            net=Path(directory)/'mapped.v';net.write_text('\n'.join(self.fixture()[:-2]))
            with self.assertRaises(ValueError):crossings(net)

    def test_port_filters_execute_without_vendor_collection_commands(self):
        with tempfile.TemporaryDirectory()as directory:
            net=Path(directory)/'mapped.v';net.write_text('\n'.join(self.fixture()))
            script='''proc all_inputs {} {return {io_coreClock io_memoryClock io_coreReset io_memoryReset io_request_valid}}
proc all_outputs {} {return {io_coreActive io_request_ready io_response_valid}}
proc get_ports {names} {return $names}
proc get_pins {names} {return $names}
foreach name {set_cmd_units create_clock set_clock_uncertainty set_input_delay set_input_transition set_output_delay set_load source simd_constrain_gray} {
 proc $name {args} {}
}
proc all_clocks {} {return {core memory}}
'''+system_sdc(net,'helper.tcl')+'''
puts $data_inputs
puts $memory_outputs
'''
            result=subprocess.run(['tclsh'],input=script,capture_output=True,text=True,check=True)
            self.assertFalse(result.stderr,result.stderr)
            self.assertEqual(result.stdout.splitlines(),[
                'io_memoryReset io_request_valid','io_request_ready io_response_valid'])


if __name__=='__main__':unittest.main()
