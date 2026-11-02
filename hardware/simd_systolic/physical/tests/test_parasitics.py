import sys
import json
import hashlib
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from parasitics import audit

GOOD='SIMD_SPEF_READ_BEGIN\nFound 0 unannotated drivers.\nFound 0 partially unannotated drivers.\nSIMD_SPEF_READ_END\n'
SPEF='*SPEF "ieee 1481-1999"\n*NAME_MAP\n*1 safe_name\n*D_NET *1 .001\n*END\n'


class ParasiticsAuditTest(unittest.TestCase):
    def check(self,log=GOOD,spef=SPEF):
        with tempfile.TemporaryDirectory()as directory:
            p=Path(directory)/'test.spef';p.write_text(spef);return audit(log,p)
    def test_complete_error_free_annotation_is_accepted(self):
        r=self.check();self.assertTrue(r['clean'],r);self.assertEqual(r['spef_nets'],1)
    def test_parser_warning_invalidates_positive_sta_numbers(self):
        r=self.check(GOOD.replace('Found 0 unannotated','[WARNING STA-1657] syntax error\nFound 0 unannotated'))
        self.assertFalse(r['clean'])
    def test_nonzero_and_partial_annotation_are_rejected(self):
        for text in('Found 1 unannotated drivers.','Found 1 partially unannotated drivers.'):
            with self.subTest(text=text):
                old='Found 0 partially unannotated drivers.'if'partially'in text else'Found 0 unannotated drivers.'
                self.assertFalse(self.check(GOOD.replace(old,text))['clean'])
    def test_absent_coverage_or_truncated_spef_is_rejected(self):
        self.assertFalse(self.check('worst slack .1\n')['clean'])
        self.assertFalse(self.check(spef=SPEF.replace('*END\n',''))['clean'])
    def test_dangling_name_escape_and_duplicate_nets_are_rejected(self):
        self.assertFalse(self.check(spef=SPEF.replace('safe_name','broken\\'))['clean'])
        self.assertFalse(self.check(spef=SPEF+'*D_NET *1 .001\n*END\n')['clean'])
    def test_zero_fanout_proof_is_exact_and_cannot_exempt_a_connected_row(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);spef=root/'x.spef';spef.write_text(SPEF)
            table=root/'drivers.tsv';proof=root/'proof.json'
            header='driver_utf8_hex\tclassification\tsinks\trouted\n'
            def save(sinks):
                table.write_text(header+'dead/Y'.encode().hex()+f'\tzero_fanout_no_routing\t{sinks}\t0\n')
                record=dict(passed=True,no_timing_exceptions=True,driver_table=str(table),
                    source_spef_sha256=hashlib.sha256(spef.read_bytes()).hexdigest(),
                    evidence={str(table):hashlib.sha256(table.read_bytes()).hexdigest()})
                proof.write_text(json.dumps(record))
            log=GOOD.replace('Found 0 unannotated drivers.','Found 1 unannotated drivers.\n dead/Y')
            save(0);self.assertTrue(audit(log,spef,proof)['clean'])
            self.assertFalse(audit(log.replace('dead/Y','connected/Y'),spef,proof)['clean'])
            save(1)
            with self.assertRaises(ValueError):audit(log,spef,proof)
            save(0);spef.write_text(SPEF.replace('.001','.002'))
            with self.assertRaises(ValueError):audit(log,spef,proof)
