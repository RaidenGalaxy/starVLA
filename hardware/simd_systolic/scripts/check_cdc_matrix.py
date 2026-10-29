#!/usr/bin/env python3
"""Digital FIFO clock/phase stress. Never claims analogue CDC signoff."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def check(out,jobs=4):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    cmd=['verilator','--cc','--exe','--build','--assert','-j',str(jobs),'--top-module','AsyncFifo',
         '-GWIDTH=32','-GADDR=4','--Mdir',str(out/'obj'),str(ROOT/'rtl/AsyncFifo.sv'),
         str(ROOT/'tests/cdc_main.cpp'),'-o',str(out/'sim')]
    with(out/'build.log').open('w')as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    cases=[]
    for wp,rp in [(4,16),(16,4),(20,74),(74,20)]:
        for phase in (0,min(7,rp//2-1)):
            run=subprocess.run([str(out/'sim'),str(wp),str(rp),str(phase)],capture_output=True,text=True,timeout=60)
            (out/f'{wp}_{rp}_{phase}.log').write_text(run.stdout+run.stderr)
            if run.returncode:raise RuntimeError(run.stderr)
            cases.append(json.loads(run.stdout))
    report=dict(passed=True,scope='digital_fifo_ratio_phase_full_empty_wrap_reset',cases=cases,
                primary_tick_ps=250,stress_tick_ps=50,physical_signoff=False,
                rtl_sha256=hashlib.sha256((ROOT/'rtl/AsyncFifo.sv').read_bytes()).hexdigest(),
                testbench_sha256=hashlib.sha256((ROOT/'tests/cdc_main.cpp').read_bytes()).hexdigest(),
                simulator_sha256=hashlib.sha256((out/'sim').read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'.build/cdc_matrix')
    parser.add_argument('-j',type=int,default=4);args=parser.parse_args();check(args.out,args.j)
