import random,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from simd_eval import cpu_reference
from simd_eval.reference import Format,gemm,fp_operation,Op

class HostFmaReferenceTest(unittest.TestCase):
    def test_all_classes_flags_and_reduction_order_match_exact_model(self):
        rng=random.Random(982)
        for fmt,width in[(Format.F4,4),(Format.F8,8),(Format.F16,16),(Format.BF16,16),(Format.F32,32)]:
            for trial in range(12):
                a=[[rng.getrandbits(width)for _ in range(11)]for _ in range(2)]
                b=[[rng.getrandbits(width)for _ in range(3)]for _ in range(11)]
                initial=[[rng.getrandbits(32)for _ in range(3)]for _ in range(2)]
                expected,flags=gemm(a,b,fmt,initial=initial,return_flags=True)
                actual,status,_=cpu_reference.gemm(a,b,fmt,initial)
                self.assertEqual((actual,status),(expected,flags))
        # Distinguish tininess at an exact midpoint from rounding to normal
        # before the subnormal precision reduction.
        for b in(0xbf000000,0xbe800000):
            expected,flags=fp_operation(1,b,0x00800000,Format.F32,Op.MAC)
            actual,status,_=cpu_reference.gemm([[1]],[[b]],Format.F32,[[0x00800000]])
            self.assertEqual((actual[0][0],status[0][0]),(expected,flags))

    def test_all_fp4_and_fp8_product_pairs(self):
        for fmt,size in[(Format.F4,16),(Format.F8,256)]:
            a=[[v]for v in range(size)];b=[list(range(size))]
            expected,flags=gemm(a,b,fmt,return_flags=True)
            actual,status,_=cpu_reference.gemm(a,b,fmt)
            self.assertEqual((actual,status),(expected,flags))
