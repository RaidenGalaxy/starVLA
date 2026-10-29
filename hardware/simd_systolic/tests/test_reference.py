import ctypes
import ctypes.util
import hashlib
import random
import struct
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from simd_eval.reference import Format, Op, decode, encode, fp_operation, operate, pack, signed, NX, UF
from simd_eval.reporting import compare,verify_timing,memory_evidence,numerical_errors,utilization_evidence,dram_evidence,sequencer_cycle_evidence,dma_link_metrics


class ArithmeticOracleTest(unittest.TestCase):
    def test_ocp_e2m1_encoding(self):
        values = [0, Fraction(1, 2), 1, Fraction(3, 2), 2, 3, 4, 6]
        for i, expected in enumerate(values):
            self.assertEqual(decode(i, Format.F4).magnitude, expected)
            self.assertEqual(decode(i + 8, Format.F4).sign, 1)
            self.assertEqual(encode(expected, Format.F4)[0], i)

    def test_wide_integer_accumulation_and_overflow(self):
        for width, fmt, aw in [(4, Format.I4, 32), (64, Format.I64, 144)]:
            low = -(1 << (width - 1))
            value, flags, overflow = operate(pack([low] * (64 // width), width),
                                             pack([low] * (64 // width), width), 0, fmt, Op.MUL)
            self.assertEqual(value & ((1 << aw) - 1), low * low)
            self.assertEqual((flags, overflow), (0, 0))
        value, _, overflow = operate(1, 1, (1 << 143) - 1, Format.I64, Op.MAC, 1)
        self.assertEqual(signed(value, 144), -(1 << 143))
        self.assertEqual(overflow, 1)

    def test_rounding_ties_and_subnormals(self):
        self.assertEqual(encode(Fraction(1) + Fraction(1, 1 << 24), Format.F32), (0x3f800000, NX))
        self.assertEqual(encode(Fraction(1) + Fraction(3, 1 << 24), Format.F32), (0x3f800002, NX))
        self.assertEqual(encode(Fraction(1, 1 << 149), Format.F32), (1, 0))

    def test_tininess_after_unbounded_exponent_rounding(self):
        tiny=Fraction(1,1<<126);ulp=Fraction(1,1<<149)
        self.assertEqual(encode(tiny-ulp/2,Format.F32),(0x800000,UF|NX))
        self.assertEqual(encode(tiny-ulp/4,Format.F32),(0x800000,NX))

    def test_fused_result_against_system_libm(self):
        # libm supplies an independent oracle for the most complex reference
        # operation. Converting bits through Python is exact for finite values.
        lib = ctypes.CDLL(ctypes.util.find_library('m'))
        rng = random.Random(651)
        for fmt, width, code, ctype, name in [(Format.F32, 32, 'f', ctypes.c_float, 'fmaf'),
                                            (Format.F64, 64, 'd', ctypes.c_double, 'fma')]:
            fma = getattr(lib, name)
            fma.argtypes = [ctype, ctype, ctype]
            fma.restype = ctype
            nbytes = width // 8
            for _ in range(400):
                bits = [rng.getrandbits(width) for _ in range(3)]
                if any(decode(v, fmt).kind != 'finite' for v in bits):
                    continue
                nums = [struct.unpack('<' + code, v.to_bytes(nbytes, 'little'))[0] for v in bits]
                actual = int.from_bytes(struct.pack('<' + code, fma(*nums)), 'little')
                expected, _ = fp_operation(*bits, fmt, Op.MAC)
                self.assertEqual(expected, actual, (fmt, bits))


class ComparisonTest(unittest.TestCase):
    def test_utilization_separates_idle_cycles_and_inactive_lanes(self):
        report=utilization_evidence('V 8 3 1\n',{'utilization_counters':True},8,16,2,4)
        self.assertEqual(report['array_issue_utilization'],0.5)
        self.assertEqual(report['lane_utilization'],0.5)
        self.assertEqual(report['mac_utilization'],0.25)
        for record in ('V 17 0 0\n','V 8 6 3\n','V 3 0 0\n',''):
            with self.assertRaises(ValueError):
                utilization_evidence(record,{'utilization_counters':True},8,16,2,4)
    def test_error_metrics_do_not_hide_underflow_or_nan(self):
        tiny=numerical_errors([[0]],[[Fraction(1,10**600)]],Format.F64)
        self.assertEqual(tiny['relative_l2'],1)
        self.assertIsNone(tiny['rmse'])
        self.assertIn('E-600',tiny['rmse_decimal'].upper())
        special=numerical_errors([[0x7ff8000000000000]],[[Fraction(0)]],Format.F64)
        self.assertIsNone(special['rmse'])
        self.assertEqual(special['nonfinite_output_count'],1)
    def report(self):
        return dict(workload_sha256='a', rtl_sha256='b', stages=12, rows=8, cols=8,
                    core_frequency_hz=1e9, memory_frequency_hz=250e6, dram=True,
                    m=7, n=19, k=65, exact_output_match=True, timing_validated=False,
                    elapsed_ps=100, format='BF16',dram_config_sha256='c',
                    simulator_sha256='d',testbench_sha256='e',dram_library_sha256='f',dma_backend='rtl',input_schedule='descriptor_sequential_gather')

    def test_comparison_requires_common_conditions(self):
        a = self.report()
        b = dict(a, elapsed_ps=50, format='I4')
        self.assertEqual(compare(a, b)['speedup'], 2)
        self.assertFalse(compare(a, b)['timing_validated'])
        for key in ('workload_sha256', 'rtl_sha256', 'core_frequency_hz', 'dram', 'simulator_sha256','dram_library_sha256','dma_backend','input_schedule'):
            with self.assertRaisesRegex(ValueError, key):
                compare(a, dict(b, **{key: 'changed'}))

    def test_sequencer_cycle_categories_cover_each_descriptor_once(self):
        caps=dict(sequencer_cycle_categories=['packing','dram_wait','core_wait'])
        jobs={1:dict(memory_cycles=15),2:dict(memory_cycles=20)}
        result=sequencer_cycle_evidence('Z 1 3 5 7\nZ 2 4 6 10\n',caps,jobs)
        self.assertEqual(result['sequencer_memory_cycles'],35)
        self.assertEqual(result['sequencer_cycle_breakdown'],dict(packing=7,dram_wait=11,core_wait=17))
        for trace in('Z 1 3 5 8\nZ 2 4 6 10\n','Z 1 3 5 7\n','Z 1 3 5 7\nZ 1 3 5 7\n','Z 1 -3 5 13\nZ 2 4 6 10\n'):
            with self.assertRaises(ValueError):sequencer_cycle_evidence(trace,caps,jobs)

    def test_dma_capacity_is_per_direction_with_independent_ports(self):
        metrics=dict(elapsed_ps=1_000_000,memory_frequency_hz=250e6)
        result=dma_link_metrics(metrics,60,40)
        self.assertEqual(result['dma_peak_per_direction_GB_s'],4)
        self.assertEqual(result['dma_transfer_bytes'],6400)
        self.assertAlmostEqual(result['dma_read_bus_utilization'],.96)
        self.assertAlmostEqual(result['dma_write_bus_utilization'],.64)
        with self.assertRaisesRegex(ValueError,'capacity'):dma_link_metrics(metrics,64,0)

    def test_memory_evidence_requires_independent_count_match(self):
        caps=self.report()
        self.assertTrue(memory_evidence('B 3 2 160 80 80 3 2\n',caps,3,2,160,80)['write_payload_byte_match'])
        self.assertTrue(memory_evidence('B 3 2 160 80 80 3 2\n',caps,0,0,160,80,False)['write_payload_byte_match'])
        with self.assertRaisesRegex(ValueError,'disagree'):
            memory_evidence('B 3 2 160 80 64 3 2\n',caps,3,2,160,80)
        observed=dict(caps,dma_overlap_observation=True)
        valid='B 3 2 160 80 80 3 2\nU 32 2\n'
        self.assertEqual(memory_evidence(valid,observed,3,2,160,80)['dma_load_bytes_while_core_active'],32)
        for record in('B 3 2 160 80 80 3 2\n',valid.replace('U 32 2','U 176 11')):
            with self.assertRaises(ValueError):memory_evidence(record,observed,3,2,160,80)

    def test_dram_geometry_distinguishes_fragments_and_coalesced_commands(self):
        split=dram_evidence('P 32 6 4\n',3,2,6,4)
        self.assertEqual(split['dram_physical_bytes'],320)
        merged=dram_evidence('P 128 4 2\n',4,2,2,1)
        self.assertEqual(merged['dram_physical_bytes'],384)
        # A forwarded read may complete without an external read command.
        self.assertEqual(dram_evidence('P 64 1 1\n',1,1,0,1)['dram_physical_bytes'],64)
        for line in ('P 32 3 2\n','P 48 6 4\n','P 32 6 4\nP 32 6 4\n'):
            with self.assertRaises(ValueError):dram_evidence(line,3,2,6,4)
        with self.assertRaises(ValueError):dram_evidence('P 64 1 1\n',1,1,2,1)

    def test_timing_claim_requires_evidence(self):
        self.assertFalse(verify_timing(None, {'rtl_sha256': 'a'}, 12, 8, 8, 1000))
        with self.assertRaisesRegex(ValueError, 'exact simulated implementation'):
            verify_timing({'timing_status': 'array_closed'}, {'rtl_sha256': 'a'}, 12, 8, 8, 1000)

    def test_timing_proof_rejects_wrong_scope_and_modified_files(self):
        with tempfile.TemporaryDirectory(prefix='simd_timing_unit_')as temporary:
            root=Path(temporary)
            proof=dict(timing_status='kernel_closed',scope='array',rows=2,cols=2,stages=12,
                period_ns=1,rtl_sha256='unit-test-only',setup_wns_ns=0.01,hold_wns_ns=0.01,tns_ns=0,
                electrical_clean=True,coverage_clean=True,route_drc_clean=True,setup_hold_closed=True,
                evidence={},evidence_sha256={})
            for name in ('netlist','sdc','spef','setup','hold','electrical','coverage','coverage_analysis','drc'):
                path=root/name;path.write_text('synthetic unit test fixture '+name)
                proof['evidence'][name]=str(path)
                proof['evidence_sha256'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
            proof.update(netlist_sha256=proof['evidence_sha256']['netlist'],sdc_sha256=proof['evidence_sha256']['sdc'])
            caps=dict(target='array',rtl_sha256='unit-test-only')
            self.assertTrue(verify_timing(proof,caps,12,2,2,1000))
            for changed in (dict(proof,scope='fp64'),dict(proof,setup_wns_ns=-0.001),dict(proof,tns_ns=-0.1)):
                with self.assertRaises(ValueError):verify_timing(changed,caps,12,2,2,1000)
            with self.assertRaises(ValueError):verify_timing(proof,dict(caps,target='system'),12,2,2,1000)
            with self.assertRaises(ValueError):verify_timing(proof,caps,12,8,8,1000)
            with self.assertRaises(ValueError):verify_timing(proof,caps,12,2,2,900)
            sdc=root/'sdc';original=sdc.read_text()
            sdc.write_text('create_clock -period 1 [get_ports clock]\nset_false_path -from [get_ports reset]\n')
            proof['sdc_sha256']=proof['evidence_sha256']['sdc']=hashlib.sha256(sdc.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'synchronous core reset'):
                verify_timing(proof,caps,12,2,2,1000)
            sdc.write_text(original)
            proof['sdc_sha256']=proof['evidence_sha256']['sdc']=hashlib.sha256(sdc.read_bytes()).hexdigest()
            (root/'spef').write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed timing evidence: spef'):
                verify_timing(proof,caps,12,2,2,1000)


if __name__ == '__main__':
    unittest.main()
