#!/usr/bin/env python3
"""Independent exact oracle for the memory-domain FP32 -> FP16/BF16 cast."""
import hashlib,json,random,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from simd_eval.reference import Format,decode,encode,canonical_nan,NV
from build import sources,rtl_digest

def run():
    out=ROOT/'.build/narrow';out.mkdir(parents=True,exist_ok=True)
    files=[ROOT/'vendor/fpnew_pkg.sv',ROOT/'rtl/lzc.sv',ROOT/'vendor/fpnew_classifier.sv',
           ROOT/'vendor/fpnew_rounding.sv',ROOT/'vendor/fpnew_cast_multi.sv',ROOT/'rtl/SimdNarrow32.sv']
    subprocess.run(['verilator','--cc','--exe','--build','-j','8','--assert','-Wno-fatal',
        '-I'+str(ROOT/'vendor'),'--top-module','SimdNarrow32','--Mdir',str(out/'obj'),
        '-o',str(out/'sim'),*map(str,files),str(ROOT/'tests/narrow_main.cpp')],check=True)
    rng=random.Random(802);count=0
    with(out/'vectors.txt').open('w')as stream:
        for fmt in(Format.F16,Format.BF16):
            values=[rng.getrandbits(32)for _ in range(20000)]
            # Every finite target encoding, including subnormals and both zeros,
            # must survive exact expansion followed by narrowing.
            for raw in range(65536):
                v=decode(raw,fmt)
                if v.kind=='finite':values.append(encode((-1 if v.sign else 1)*v.magnitude,Format.F32,v.sign)[0])
            for value in values:
                v=decode(value,Format.F32)
                if v.kind=='nan':expected,flags=canonical_nan(fmt),NV if v.signaling else 0
                elif v.kind=='inf':expected,flags=((v.sign<<15)|(0x7f80 if fmt==Format.BF16 else 0x7c00)),0
                else:expected,flags=encode((-1 if v.sign else 1)*v.magnitude,fmt,v.sign)
                stream.write(f'{value:x} {int(fmt==Format.BF16):x} {expected:x} {flags:x}\n');count+=1
    subprocess.run([str(out/'sim'),str(out/'vectors.txt')],check=True)
    report=dict(passed=True,vectors=count,formats=['F16','BF16'],rtl_sha256=rtl_digest(files),
                simulator_sha256=hashlib.sha256((out/'sim').read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':run()
