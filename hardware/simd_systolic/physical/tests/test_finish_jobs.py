import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from finish_migration_jobs import finalize_execution_log


class FinalExecutionLogTest(unittest.TestCase):
    def case(self,changed_prefix=False,extra_suffix=False):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);log=root/'execution.log';path=root/'bounded_execution.json'
            prefix=b'actual tool result\n'
            record=dict(returncode=0,execution_log=str(log),log_sha256=hashlib.sha256(prefix).hexdigest())
            original=json.dumps(record,indent=2)+'\n';path.write_text(original)
            log.write_bytes((b'changed result\n'if changed_prefix else prefix)+original.encode()+
                (b'extra output\n'if extra_suffix else b''))
            if changed_prefix or extra_suffix:
                with self.assertRaises(ValueError):finalize_execution_log(path,record)
                self.assertEqual(path.read_text(),original)
                self.assertFalse((root/'bounded_execution.unfinalized.json').exists())
            else:
                result=finalize_execution_log(path,record)
                self.assertEqual(result['log_sha256'],hashlib.sha256(log.read_bytes()).hexdigest())
                self.assertEqual((root/'bounded_execution.unfinalized.json').read_text(),original)
                self.assertEqual(result['returncode'],record['returncode'])

    def test_exact_producer_record_suffix_finalizes_after_exit(self):self.case()
    def test_changed_previously_hashed_result_is_rejected(self):self.case(changed_prefix=True)
    def test_unexpected_extra_output_is_rejected(self):self.case(extra_suffix=True)
