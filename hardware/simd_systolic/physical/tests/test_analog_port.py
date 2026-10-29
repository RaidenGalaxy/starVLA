import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_analog import port_annotations
from check_sram_bitcell import model_port


class AnalogCompilerPortTest(unittest.TestCase):
    def test_quoted_parentheses_do_not_change_equations(self):
        source='parameter real X = 2*Y `attr(info="physical (equation)");\n'
        self.assertEqual(port_annotations(source),'parameter real X = 2*Y ;\n')

    def test_instance_annotation_moves_before_declaration(self):
        source='parameter real NFIN = 1 from (0:inf) `attr(type="instance" xyceAlsoModel="YES" info="fins (count)");\n'
        self.assertEqual(port_annotations(source),'(* type="instance" *) parameter real NFIN = 1 from (0:inf) ;\n')

    def test_physical_model_parameters_are_unchanged(self):
        source=''.join(f'.model model_{i} '+('nmos'if i<4 else'pmos')+
            ' level = 72\n+version=107 eotacc=1e-10 nfin=2\n'for i in range(8))
        converted=model_port(source)
        self.assertEqual(converted.count('eotacc=1e-10 nfin=2'),8)
        self.assertEqual(converted.count('devtype=1'),4)
        self.assertEqual(converted.count('devtype=0'),4)

    def test_other_bsim_version_fails_closed(self):
        source=''.join(f'.model model_{i} nmos level=72\n+version=110 eotacc=1e-10\n'for i in range(8))
        with self.assertRaises(ValueError):model_port(source)


if __name__=='__main__':unittest.main()
