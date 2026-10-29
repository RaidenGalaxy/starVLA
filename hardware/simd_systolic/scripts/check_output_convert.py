#!/usr/bin/env python3
"""Check output RNE, FP4 finite saturation and signed zero-point boundaries."""
import argparse,hashlib,json,random,shutil,subprocess,sys
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from simd_eval.reference import Format,to_bits
from simd_eval.epilogue import convert32,floating_cast


def check(out,jobs):
    out.mkdir(parents=True,exist_ok=True);snapshot=out/'inputs';snapshot.mkdir(exist_ok=True)
    inputs=[ROOT/'vendor'/name for name in ('fpnew_pkg.sv','fpnew_classifier.sv','fpnew_rounding.sv','fpnew_cast_multi.sv')]
    inputs += [ROOT/'rtl/lzc.sv',ROOT/'rtl/OutputConvert.sv',ROOT/'tests/output_convert_main.cpp']
    for p in inputs:shutil.copy2(p,snapshot/p.name)
    shutil.copytree(ROOT/'vendor/common_cells',snapshot/'common_cells',dirs_exist_ok=True)
    with(out/'build.log').open('w')as log:
        subprocess.run(['verilator','--cc','--exe','--build','-j',str(jobs),'-Wno-fatal','--top-module','SimdOutputConvert',
            '--Mdir',str(out/'obj'),'-I'+str(snapshot),*[str(snapshot/p.name)for p in inputs],'-o',str(out/'sim')],check=True,stdout=log,stderr=subprocess.STDOUT)
    edges={0,1,0x7fffff,0x800000,0x3f000000,0x3f800000,0x7f7fffff,0x7f800000,0x7f800001,0x7fc00001}
    for value in [Fraction(1,4),Fraction(3,4),Fraction(5,4),Fraction(7,4),Fraction(5,2),Fraction(7,2),Fraction(5),Fraction(6),
                  Fraction(127,2),Fraction(255,2),Fraction(65535,2),Fraction(1<<31),Fraction(1<<63),Fraction(1<<64)]:
        bits=to_bits(value,Format.F32);edges.update([bits-1,bits,bits+1])
    edges|={bits|0x80000000 for bits in list(edges)}
    rng=random.Random(601);values=sorted(edges)+[rng.getrandbits(32)for _ in range(2000)]
    zero_points=[0,1,-1,7,-8,(1<<31)-1,-(1<<31)]
    vectors=[]
    for fmt in Format:
        for bits in values:
            points=zero_points if fmt<Format.F4 and bits in edges else[0]
            for zero in points:
                result,flags,saturated=convert32(bits,fmt,zero)
                vectors.append(f'{bits:x} 0 {int(fmt):x} {zero&0xffffffff:x} {result:x} {flags:x} {int(saturated):x}\n')
    for bits in [0,1,0xfffffffffffff,0x10000000000000,0x7fefffffffffffff,0x7ff0000000000000,0x7ff0000000000001,
                 0xfff8000000000000]+[rng.getrandbits(64)for _ in range(4000)]:
        result,flags=floating_cast(bits,Format.F64,Format.F32)
        vectors.append(f'{bits:x} 1 9 0 {result:x} {flags:x} 0\n')
    trace=out/'vectors.txt';trace.write_text(''.join(vectors))
    run=subprocess.run([str(out/'sim'),str(trace)],check=True,text=True,capture_output=True,timeout=120)
    report=json.loads(run.stdout);report.update(scope='output_conversion',all_formats=True,fp64_to_fp32=True,
        integer_rne_zero_point_before_saturation=True,fp4_midpoint_neighbors=True,
        source_sha256={p.name:hashlib.sha256((snapshot/p.name).read_bytes()).hexdigest()for p in inputs},
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),simulator_sha256=hashlib.sha256((out/'sim').read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'.build/output_convert')
    parser.add_argument('-j',type=int,default=2);args=parser.parse_args();check(args.out.resolve(),args.j)
