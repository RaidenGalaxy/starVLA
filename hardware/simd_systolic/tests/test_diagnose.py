import contextlib,importlib.util,io,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('physical_diagnose',ROOT/'physical/diagnose.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class TimingPathResolutionTest(unittest.TestCase):
    def test_routed_flattened_instance_names_can_contain_slashes(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);(root/'build_identity.json').write_text(json.dumps({'top':'Top'}))
            (root/'routed.v').write_text('module Top();\nDFF_X1 \\leaf/_1_ (.Q(flat_state),.D(input_data));\nendmodule\n')
            (root/'setup.rpt').write_text('Startpoint: leaf/_1_ (flip-flop)\nEndpoint: leaf/_1_ (flip-flop)\n-0.010 slack (VIOLATED)\n')
            with contextlib.redirect_stdout(io.StringIO()):module.diagnose(root,routed=True)
            result=json.loads((root/'routed_path_diagnosis.json').read_text())['paths'][0]
            self.assertEqual(result['startpoint_state']['q'],'flat_state')

    def test_repeated_local_cell_names_resolve_in_their_own_module(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary)
            (root/'build_identity.json').write_text(json.dumps({'top':'Top'}))
            (root/'synth.v').write_text('''module Top();
 DFF_X1 _1_ (.Q(top_state),.D(top_data));
 Helper leaf ();
endmodule
module Helper();
 DFF_X2 _1_ (.Q(local_state),.D(local_data));
endmodule
''')
            (root/'preopt_setup.rpt').write_text('Startpoint: leaf/_1_ (flip-flop)\nEndpoint: _1_ (flip-flop)\n0.100 slack (MET)\n')
            with contextlib.redirect_stdout(io.StringIO()):module.diagnose(root)
            result=json.loads((root/'path_diagnosis.json').read_text())['paths'][0]
            self.assertEqual(result['startpoint_state']['q'],'local_state')
            self.assertEqual(result['endpoint_state']['q'],'top_state')
