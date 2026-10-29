#!/usr/bin/env python3
"""Authenticate standalone RTL DMA simulations and retain randomized evidence."""
import argparse,json,subprocess
from pathlib import Path
from runtime import validated_build
ROOT=Path(__file__).resolve().parents[1]


def check(kind):
    directory=ROOT/f'.build/{kind}_2_1x1';guard,caps=validated_build(directory)
    proc=subprocess.run([str(directory/'sim')],check=True,capture_output=True,text=True,timeout=120)
    report=json.loads(proc.stdout)
    if report.get('passed')is not True:raise AssertionError(report)
    report.update(kind=kind,rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'],
                  testbench_sha256=caps['testbench_sha256'])
    (directory/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['dma','write_dma','both'],default='both');a=p.parse_args()
    for kind in (['dma','write_dma']if a.kind=='both'else[a.kind]):check(kind)
