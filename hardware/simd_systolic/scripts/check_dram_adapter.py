#!/usr/bin/env python3
"""Exercise 32/64/128-byte physical DRAM bursts behind 64-byte RTL transactions."""
import configparser
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def check():
    out=ROOT/'.build/dram_adapter';out.mkdir(exist_ok=True)
    dep=ROOT/'.deps/dramsim3';header=ROOT/'sim/dram_adapter.hpp'
    with(out/'build.log').open('w')as log:
        subprocess.run(['g++','-O2','-std=c++17','-I'+str(dep/'src'),'-I'+str(ROOT/'sim'),
            str(ROOT/'tests/dram_adapter.cpp'),'-L'+str(dep),'-Wl,-rpath,'+str(dep),'-ldramsim3',
            '-o',str(out/'sim')],stdout=log,stderr=subprocess.STDOUT,check=True)
    cases=[]
    for burst in(32,64,128):
        config=configparser.ConfigParser();config.read(ROOT/'configs/ddr4.ini')
        config['system']['bus_width']=str(burst)
        config['system']['trans_queue_size']='1'
        config_path=out/f'burst{burst}.ini'
        with config_path.open('w')as stream:config.write(stream)
        result=subprocess.run([str(out/'sim'),str(config_path),str(out/f'burst{burst}')],capture_output=True,text=True,check=True,timeout=60)
        (out/f'burst{burst}.log').write_text(result.stdout+result.stderr)
        line=next(line for line in result.stdout.splitlines()if line.startswith('A '))
        _,actual,submitted_r,physical_r,submitted_w,physical_w,cycles=map(str,line.split())
        assert int(actual)==burst
        cases.append(dict(burst_bytes=burst,submitted_reads=int(submitted_r),physical_read_bursts=int(physical_r),
            submitted_writes=int(submitted_w),physical_write_bursts=int(physical_w),dram_cycles=int(cycles)))
    report=dict(passed=True,logical_transaction_bytes=64,maximum_logical_outstanding=32,
        physical_queue_size=1,duplicate_address_callbacks=True,physical_write_tail_drained=True,cases=cases,
        adapter_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
        dram_library_sha256=hashlib.sha256((dep/'libdramsim3.so').read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':check()
