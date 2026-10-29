import hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from simd_eval.area import load_area,area_metrics


class AreaEvidenceTest(unittest.TestCase):
    def test_area_requires_matching_scope_netlist_and_cell_statistics(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);net=root/'synth.v';net.write_text('module gemm; endmodule\n')
            library=root/'cells.lib';library.write_text('library(test) {}\n')
            statistics=root/'cell_statistics.json';statistics.write_text(json.dumps({'design':{'area':1200}}))
            identity=dict(scope='gemm',stages=2,rows=2,cols=2,rtl_sha256='rtl')
            report=dict(identity,evidence_kind='standard_cell_synthesis',cell_area_um2=1200,
                netlist=str(net),mapped_sha256=hashlib.sha256(net.read_bytes()).hexdigest(),
                liberty_file=str(library),liberty_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
            path=root/'synthesis.json';path.write_text(json.dumps(report));proof=load_area(path)
            result=area_metrics(dict(identity,gmac_s=3),proof,'gemm')
            self.assertEqual(result['cell_area_um2'],1200)
            self.assertEqual(result['area_scope'],'gemm_system')
            self.assertFalse(result['area_timing_certified'])
            for changed,scope in[(dict(identity,rtl_sha256='different'),'gemm'),(identity,'array'),(dict(identity,rows=8),'gemm')]:
                with self.assertRaisesRegex(ValueError,'exact simulated'):area_metrics(changed,proof,scope)
            statistics.write_text(json.dumps({'design':{'area':2400}}))
            with self.assertRaisesRegex(ValueError,'statistics'):load_area(path)
            net.write_text('module altered; endmodule\n')
            with self.assertRaisesRegex(ValueError,'netlist changed'):load_area(path)
