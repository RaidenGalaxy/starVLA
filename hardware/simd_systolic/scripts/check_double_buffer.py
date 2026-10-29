#!/usr/bin/env python3
"""Verify real DMA overlap between the next SRAM tile and the active FIFO tile."""
import argparse
import json
from pathlib import Path
from runtime import validated_build
from run_gemm import run

ROOT=Path(__file__).resolve().parents[1]
import sys as _architecture_sys
_architecture_sys.path.insert(0,str(ROOT/'python'))
from simd_eval.architecture import CORE_PS, MEMORY_PS


def check(args):
    build=args.build_dir.resolve();guard,caps=validated_build(build)
    if caps['target']!='system'or not caps.get('dma_overlap_observation'):
        raise ValueError('rebuild a TileSystem simulator with DMA/core overlap observations')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True);reports={}
    for serialized in(True,False):
        name='barrier'if serialized else'overlap'
        reports[name]=run(fmt='I8',m=caps['rows'],n=caps['cols']*8*16,k=24,
            stages=caps['stages'],rows=caps['rows'],cols=caps['cols'],dram=args.dram,
            core_ps=CORE_PS,memory_ps=MEMORY_PS,out=out/name,serialize_loads=serialized,build_dir=build)
    baseline,candidate=reports['barrier'],reports['overlap']
    for key in('rtl_sha256','simulator_sha256','testbench_sha256','dram_library_sha256',
               'macs','compute_cycles','dma_payload_bytes','output_payload_bytes','dram_physical_bytes'):
        if baseline[key]!=candidate[key]:raise AssertionError('buffer ablation changed '+key)
    if baseline['dma_load_bytes_while_core_active']!=0:raise AssertionError('barrier schedule still overlaps loads')
    if candidate['dma_load_bytes_while_core_active']<=0:raise AssertionError('no next-tile data committed while the core was active')
    report=dict(passed=True,kind='input_double_buffer_ablation',stages=caps['stages'],rows=caps['rows'],cols=caps['cols'],
        organization='current tile in wave FIFO; next tile in A/B SRAM',dram=args.dram,
        noninteger_clock_ratio=False,core_period_ps=CORE_PS,memory_period_ps=MEMORY_PS,
        barrier_elapsed_ps=baseline['elapsed_ps'],overlap_elapsed_ps=candidate['elapsed_ps'],
        scheduling_speedup=baseline['elapsed_ps']/candidate['elapsed_ps'],
        overlapped_dma_load_bytes=candidate['dma_load_bytes_while_core_active'],
        identical_outputs_and_compute_cycles=True,autonomous_gemm_prefetch=False,
        time_kind='configured_clock_projection',timing_validated=False,
        rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'])
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--dram',action='store_true');parser.add_argument('--out',type=Path,required=True)
    check(parser.parse_args())
