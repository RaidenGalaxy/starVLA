import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from simd_eval.functional_coverage import FunctionalCoverage,bin_spec,numerical_bins
from simd_eval.hardware_contract import hardware_contract
from simd_eval.reporting import dram_geometry,runtime_clock_metrics
from simd_eval.validation import require_output_bytes


class VerificationAcceptanceTest(unittest.TestCase):
    def test_missing_critical_cannot_hide_in_99_percent(self):
        bins=[bin_spec(str(i),'test',i==99)for i in range(100)]
        coverage=FunctionalCoverage({'rtl_sha256':'frozen'},bins)
        coverage.checked([str(i)for i in range(99)],witness='observed',actual=1,expected=1)
        self.assertEqual(coverage.report()['functional_coverage_percent'],99)
        self.assertFalse(coverage.report()['passed'])
        with self.assertRaises(ValueError):coverage.checked(['unknown'],witness='x',actual=0,expected=0)
        with self.assertRaises(AssertionError):coverage.checked(['99'],witness='bad',actual=1,expected=0)
        coverage.checked(['99'],witness='later_pass',actual=0,expected=0)
        self.assertFalse(coverage.report()['passed'])

    def test_configuration_and_bin_identity_are_bound(self):
        a=FunctionalCoverage({'rtl_sha256':'a'},numerical_bins('pe'))
        b=FunctionalCoverage({'rtl_sha256':'b'},numerical_bins('pe'))
        self.assertNotEqual(a.model_sha256,b.model_sha256)
        self.assertEqual(a.report()['hit_bins'],0)

    def test_ports_and_true_wide_results(self):
        result=hardware_contract(dict(target='gemm',rows=8,cols=8))
        formats={x['format']:x for x in result['formats']}
        self.assertEqual((formats['I4']['lanes'],formats['I4']['accumulator_bits']),(16,32))
        self.assertEqual(formats['I32']['result_bits'],160)
        self.assertEqual(formats['I64']['result_bits'],144)
        self.assertEqual((result['active_read_max_outstanding'],result['standalone_read_dma_max_outstanding']),(1,32))
        self.assertIn('FP32',formats['F4']['implementation'])
        alu=hardware_contract(dict(target='alu',rows=1,cols=1))
        self.assertEqual(alu['context_logical_read_ports'],0)
        self.assertFalse(alu['contains_pe_context_state'])

    def test_three_clocks_and_posted_tail(self):
        result=dram_geometry('K 630 64 8\nP 64 12 4\n')
        self.assertAlmostEqual(result['dram_frequency_hz'],1e12/630)
        self.assertAlmostEqual(result['dram_peak_shared_bus_GB_s'],25.3968253968)
        clocks=runtime_clock_metrics(dict(core_frequency_hz=1e9,memory_frequency_hz=250e6,elapsed_ps=10000,submitted_ps=8000))
        self.assertEqual(clocks['memory_period_ps'],4000)
        self.assertEqual(clocks['dma_write_peak_GB_s'],4)
        self.assertEqual(clocks['final_drain_tail_ps'],2000)
        for text in ['K 630 32 8\nP 64 1 1\n','K 630 64 8\nK 625 64 8\nP 64 1 1\n']:
            with self.assertRaises(ValueError):dram_geometry(text)
        self.assertFalse(dram_geometry('P 64 1 1\n')['dram_runtime_geometry_observed'])

    def test_high_carry_truncation_and_missing_zero_byte_are_rejected(self):
        coverage=FunctionalCoverage({'rtl_sha256':'bound'},[bin_spec('wide','wide')])
        value=1<<142
        with self.assertRaises(AssertionError):
            coverage.checked(['wide'],witness={'tag':65535},actual=value&((1<<128)-1),expected=value)
        self.assertEqual(coverage.report()['hit_bins'],0)
        with self.assertRaisesRegex(AssertionError,'output byte'):
            require_output_bytes({0:42},{0:42,1:0})
        with self.assertRaisesRegex(ValueError,'beats'):
            from simd_eval.reporting import dram_evidence
            dram_evidence('P 64 1 1\nPORTS 1 1 4 3\n',1,1,1,1)


if __name__=='__main__':unittest.main()
