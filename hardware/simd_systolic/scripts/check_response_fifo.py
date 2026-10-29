#!/usr/bin/env python3
"""Verify registered queue heads against independent software deques."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def check(jobs=2,build_dir=None):
    work=Path(build_dir).resolve()if build_dir else ROOT/'.build/response_fifo'
    work.mkdir(parents=True,exist_ok=True)
    inputs=[ROOT/'rtl/ControlReplicas.sv',ROOT/'tests/ResponseFifoProbe.sv',ROOT/'tests/response_fifo_main.cpp']
    for p in inputs:shutil.copy2(p,work/p.name)
    with(work/'build.log').open('w')as log:
        subprocess.run(['verilator','--cc','--exe','--build','-j',str(jobs),'-Wno-fatal','--top-module','ResponseFifoProbe',
            '--Mdir',str(work/'obj'),str(work/inputs[0].name),str(work/inputs[1].name),str(work/inputs[2].name),
            '-o',str(work/'sim')],check=True,stdout=log,stderr=subprocess.STDOUT)
    result=subprocess.run([str(work/'sim')],check=True,capture_output=True,text=True,timeout=60)
    report=json.loads(result.stdout);report['scope']='response_fifo_registered_head'
    report['source_sha256']={p.name:hashlib.sha256((work/p.name).read_bytes()).hexdigest()for p in inputs}
    report['simulator_sha256']=hashlib.sha256((work/'sim').read_bytes()).hexdigest()
    (work/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('-j',type=int,default=2)
    parser.add_argument('--build-dir',type=Path);args=parser.parse_args();check(args.j,args.build_dir)
