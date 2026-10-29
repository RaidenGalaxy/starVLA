import sys
import unittest
from fractions import Fraction
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from simd_eval.reference import Format
from simd_eval.reporting import numerical_errors,epilogue_evidence
from simd_eval.epilogue import output_configuration


class OutputEvidenceTest(unittest.TestCase):
    def test_integer_errors_reconstruct_original_values(self):
        # Signed nibble 8 is -8; remove zero point before inverse scaling.
        report=numerical_errors([[7,8]],[[Fraction(3),Fraction(-9,2)]],
                                Format.I4,scale=Fraction(2),zero_point=1)
        self.assertEqual(report['rmse'],0)
        self.assertEqual(report['relative_l2'],0)
        with self.assertRaises(ValueError):
            numerical_errors([[0]],[[0]],Format.I4,scale=0)
        config=output_configuration({'output_epilogue':True},'I4','1/3',-2)
        self.assertEqual(config['output_scale_bits'],0x3eaaaaab)
        self.assertEqual(Fraction(config['output_scale_exact']),Fraction(11184811,33554432))
        for scale in ('0','1e100','1e-100'):
            with self.assertRaises(ValueError):output_configuration({'output_epilogue':True},'I4',scale)

    def test_missing_or_fabricated_conversion_cost_rejected(self):
        caps={'output_epilogue':True}
        job=lambda:{7:dict(m=1,n=3,expected_output_saturations=1)}
        result=epilogue_evidence('Y 7 3 2 21 1\n',caps,job(),True)
        self.assertEqual(result['epilogue_elements'],3)
        for bad in ('','Y 7 2 2 21 1\n','Y 7 3 0 21 1\n','Y 7 3 2 1 1\n',
                    'Y 7 3 2 21 0\n','Y 7 3 2 21 1\nY 7 3 2 21 1\n'):
            with self.assertRaises(ValueError):epilogue_evidence(bad,caps,job(),True)
        self.assertEqual(epilogue_evidence('Y 7 0 0 0 0\n',caps,{7:dict(m=1,n=3)})['epilogue_elements'],0)


if __name__=='__main__':unittest.main()
