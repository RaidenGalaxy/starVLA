#!/usr/bin/env python3
"""Sequential equivalence of actual controller netlists through Liberty models."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(args):
    out=args.out.resolve()
    if out.exists():raise ValueError('preserve existing proof; choose a new --out')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'proof_driver.py')
    original=out/'original.v';candidate=out/'candidate.v'
    shutil.copy2(args.original,original);shutil.copy2(args.candidate,candidate)
    libs=sorted((ROOT/'.deps/openfinram/tech/lib').glob('*_RVT_TT.lib'))
    script=out/'proof.ys'
    text=''.join('read_liberty -ignore_miss_func '+str(p)+'\n'for p in libs)
    text+=f'''read_verilog {original}
rename ctrl_decode gold
read_verilog {candidate}
rename ctrl_decode gate
proc
flatten gold gate
opt_clean
equiv_make gold gate equivalent
hierarchy -check -top equivalent
opt_clean
equiv_simple -seq 2
equiv_induct -seq 4
equiv_status -assert
'''
    script.write_text(text)
    with(out/'proof.log').open('w')as log:
        process=subprocess.run(['yosys','-Q','-T','-s',str(script)],stdout=log,stderr=subprocess.STDOUT,timeout=180)
    raw=(out/'proof.log').read_text()
    report=dict(kind='actual_sram_controller_sequential_equivalence',
        passed=process.returncode==0 and 'Equivalence successfully proven!'in raw,
        returncode=process.returncode,zero_delay_library_logic=True,
        all_programmable_modes_compared=True,unchanged_controller_latency=True,
        physical_frequency_certified=False,whole_sram_transistor_behavior_verified=False,
        evidence={str(p):sha(p)for p in [*libs,*[p for p in out.iterdir()if p.is_file()]]})
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k!='evidence'},indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
