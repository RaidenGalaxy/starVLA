#!/usr/bin/env python3
"""Produce the documented encoded checkpoint interchange format from layer NPZ."""
import argparse,json,hashlib,sys
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from simd_eval.reference import Format
from simd_eval.quantization import quantize

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--format',choices=['I4','I8','F4','F8'],required=True)
    p.add_argument('--group-size',type=int,default=32);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    data=np.load(args.input,allow_pickle=False);a=data['a'];b=data['b']
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0]or args.group_size<1:raise ValueError('invalid matrices/group size')
    convert=lambda x:[[Fraction(float(v))for v in row]for row in x]
    qa,qb,sa,sb=quantize(convert(a),convert(b),Format[args.format],args.group_size)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.out,a=a,b=b,qa=np.asarray(qa,dtype=np.uint8),qb=np.asarray(qb,dtype=np.uint8),
             sa_bits=np.asarray(sa,dtype=np.uint32),sb_bits=np.asarray(sb,dtype=np.uint32),
             format=np.asarray(args.format),group_size=np.asarray(args.group_size,dtype=np.uint32))
    provenance={'source':str(args.input.resolve()),'source_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'producer':'simd_eval.quantization.quantize','format':args.format,'group_size':args.group_size,
                'encoding':'unsigned raw element codes; FP32 scale bit patterns',
                'checkpoint_sha256':hashlib.sha256(args.out.read_bytes()).hexdigest()}
    args.out.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps(provenance,indent=2))
