#!/usr/bin/env python3
"""Prove small odd/even instances of the actual experimental multiplier.

This checks radix-4 sign extension and correction bits for every input at the
listed widths. The production-width numerical check remains a separate result.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from run import sha, tool


def run(directory):
    directory=directory.resolve()
    identity=json.loads((directory/'build_identity.json').read_text())
    if identity.get('experiment')!='booth':raise ValueError('requires an actual Booth experiment')
    source=directory/'inputs/PrefixAdd.sv'
    if sha(source)!=identity['input_manifest']['PrefixAdd.sv']:
        raise ValueError('experimental multiplier input changed')
    out=directory/'booth_formal'
    out.mkdir(exist_ok=False)
    shutil.copy2(__file__,out/'check_driver.py')
    records=[]
    for width in (2,3,4,7,8):
        miter=out/f'miter_{width}.sv'
        miter.write_text(f'''module MultiplierMiter(input [{width-1}:0] a,b,output equivalent);
wire [{2*width-1}:0] actual,expected;
SimdBoothUnsignedMultiply #(.W({width})) dut(.a(a),.b(b),.result(actual));
assign expected=a*b;
assign equivalent=(actual==expected);
endmodule
''')
        script=out/f'proof_{width}.ys'
        script.write_text(f'''read_slang --top MultiplierMiter {source} {miter}
prep -top MultiplierMiter -flatten
sat -prove equivalent 1 -verify
''')
        log=out/f'proof_{width}.log'
        with log.open('w')as stream:
            result=subprocess.run([tool('yosys'),'-Q','-T','-s',str(script)],stdout=stream,stderr=subprocess.STDOUT)
        records.append(dict(width=width,all_input_pairs=2**(2*width),returncode=result.returncode,
            log=str(log),log_sha256=sha(log),script_sha256=sha(script),miter_sha256=sha(miter)))
        if result.returncode:break
    report=dict(kind='small_width_booth_exhaustive_formal_equivalence',
        passed=len(records)==5 and all(x['returncode']==0 for x in records),
        multiplier_source_sha256=sha(source),rtl_sha256=identity['rtl_sha256'],
        production_width_formally_proven=False,checks=records,driver_sha256=sha(__file__))
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise RuntimeError('Booth identity proof failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    run(p.parse_args().directory)
