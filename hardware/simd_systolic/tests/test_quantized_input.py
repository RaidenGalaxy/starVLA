import sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from simd_eval.quantization import load_codes
from simd_eval.reference import Format


class EncodedCheckpointTest(unittest.TestCase):
    def test_import_preserves_codes_instead_of_requantizing_originals(self):
        values=dict(a=np.zeros((1,3)),b=np.zeros((3,2)),qa=np.array([[0,1,15]],dtype=np.uint8),
                    qb=np.array([[1,2],[3,4],[5,6]],dtype=np.uint8),
                    sa_bits=np.array([[0x3f000000,0x40000000]],dtype=np.uint32),
                    sb_bits=np.array([[0,0x3f800000],[0x3f800000,0x3f800000]],dtype=np.uint32),
                    format=np.asarray('I4'),group_size=np.asarray(2))
        with tempfile.TemporaryDirectory()as temp:
            path=Path(temp)/'codes.npz';np.savez(path,**values)
            a,b,qa,qb,sa,sb,group=load_codes(path,Format.I4)
            self.assertEqual(qa,[[0,1,15]])
            self.assertEqual(sa,[[0x3f000000,0x40000000]])
            self.assertEqual(group,2)
            with self.assertRaisesRegex(ValueError,'format'):load_codes(path,Format.I8)
            for key,value,message in [('sb_bits',np.zeros((1,2),dtype=np.uint32),'shapes'),
                    ('sa_bits',np.array([[0x7fc00000,0]],dtype=np.uint32),'finite'),
                    ('qa',np.array([[0,-1,2]],dtype=np.int32),'unsigned')]:
                np.savez(path,**dict(values,**{key:value}))
                with self.assertRaisesRegex(ValueError,message):load_codes(path,Format.I4)
