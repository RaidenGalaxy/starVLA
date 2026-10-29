import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from simd_eval.reporting import dram_geometry,native_prefetch_evidence


class PrefetchEvidenceTest(unittest.TestCase):
    def test_dram_geometry_and_actual_overlap_remain_independent(self):
        stdout='N 7 2 256 128 96\nP 64 12 4\n'
        jobs={7:{}};caps=dict(autonomous_prefetch=True,native_prefetch_record_prefix='N')
        self.assertEqual(dram_geometry(stdout)['dram_submitted_read_transactions'],12)
        report=native_prefetch_evidence(stdout,caps,jobs)
        self.assertEqual(report['prefetch_load_bytes_while_core_active'],96)
        self.assertEqual(jobs[7]['prefetch_load_bytes_during_tile_inflight'],128)
        for invalid in ['N 7 2 256 128 300\n','N 7 2 256 300 96\n','N 7 2 256 128 96\nN 7 2 256 128 96\n','N 8 2 256 128 96\n','']:
            with self.assertRaises(ValueError):native_prefetch_evidence(invalid,caps,{7:{}})


if __name__=='__main__':unittest.main()
