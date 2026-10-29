import hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_code_coverage import summarize


class ActualCodeCountersTest(unittest.TestCase):
    def test_counter_types_and_cross_build_rejection(self):
        with tempfile.TemporaryDirectory()as td:
            path=Path(td)/'coverage.dat'
            path.write_text("# SystemC::Coverage-3\nC '\x01f\x02project.sv\x01l\x0210\x01t\x02branch\x01page\x02v_branch/top' 0\n"+
                            "C '\x01f\x02fpnew_rounding.sv\x01l\x0220\x01page\x02v_line/round' 4\n")
            caps=dict(rtl_sha256='rtl',simulator_sha256='sim')
            proof=dict(caps,coverage_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),functional_run_passed=True)
            path.with_name('coverage.dat.identity.json').write_text(json.dumps(proof))
            report=summarize([path],caps)
            self.assertEqual(report['groups']['project_rtl/branch']['percent'],0)
            self.assertEqual(report['groups']['third_party_arithmetic/line']['percent'],100)
            with self.assertRaises(ValueError):summarize([path],dict(caps,rtl_sha256='other'))
            path.write_text(path.read_text().replace(" 4\n"," 5\n"))
            with self.assertRaises(ValueError):summarize([path],caps)


if __name__=='__main__':unittest.main()
