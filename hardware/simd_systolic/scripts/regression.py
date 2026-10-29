#!/usr/bin/env python3
"""Executable regression entry point; successful reports are retained."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import sys as _architecture_sys
_architecture_sys.path.insert(0,str(ROOT/'python'))
from simd_eval.architecture import CORE_PS, MEMORY_PS


def run(args):
    def call(*cmd):
        subprocess.run(list(map(str, cmd)), cwd=ROOT, check=True)
    call(sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v')
    call(sys.executable,'scripts/check_sram_bank.py')
    call(sys.executable,'scripts/check_narrow.py')
    call(sys.executable,'scripts/check_output_convert.py','-j',args.jobs)
    call(sys.executable,'scripts/check_expansion.py')
    call('verilator', '--cc', '--exe', '--build', '--assert', '-j', args.jobs, '--top-module', 'AsyncFifo',
         '-GWIDTH=32', '-GADDR=4', '--Mdir', ROOT / '.build/cdc',
         ROOT / 'rtl/AsyncFifo.sv', ROOT / 'tests/cdc_main.cpp', '-o', ROOT / '.build/cdc/sim')
    call(ROOT / '.build/cdc/sim')
    call(sys.executable,'scripts/check_cdc.py')
    call(sys.executable,'scripts/check_synchronizers.py')
    call(sys.executable,'scripts/check_dram_adapter.py')
    for kind in ('dma','write_dma'):
        call(sys.executable,'scripts/build.py','--kind',kind,'-j',args.jobs)
    call(sys.executable,'scripts/check_dma.py')
    for stages in args.stages:
        for kind, rows in [('alu', 1),('pe',1), ('array', 8), ('system', 2),('gemm',2)]:
            flags=['--hierarchical']if kind in('pe','array','system','gemm')else[]
            call(sys.executable, 'scripts/build.py', '--kind', kind, '--stages', stages, '--rows', rows, '-j', args.jobs,*flags)
        call(sys.executable, 'scripts/check_alu.py', '--stages', stages, '--count', 200)
        call(sys.executable, 'scripts/check_alu.py', '--stages', stages, '--count', 50, '--stall', '--exhaustive')
        call(sys.executable,'scripts/check_pe.py','--stages',stages)
        call(sys.executable,'scripts/check_pe_state.py','--build-dir',ROOT/f'.build/pe_{stages}_1x1')
        if stages>2:call(sys.executable,'scripts/check_pe_reset.py','--build-dir',ROOT/f'.build/pe_{stages}_1x1')
        call(sys.executable,'scripts/check_ii.py','--stages',stages)
        call(sys.executable,'scripts/check_pe_long_k.py','--stages',stages)
        call(sys.executable, 'scripts/check_array.py', '--stages', stages, '--rows', 8, '--cols', 8, '--pipelined-read')
        call(sys.executable,'scripts/check_array_cycles.py','--build-dir',ROOT/f'.build/array_{stages}_8x8')
        call(sys.executable,'scripts/check_cycle_stress.py','--build-dir',ROOT/f'.build/array_{stages}_8x8',
             '--out',ROOT/f'artifacts/mesh_cycle_stress_regression_{stages}_8x8')
        call(sys.executable,'scripts/check_array_hazard.py','--build-dir',ROOT/f'.build/array_{stages}_8x8')
        call(sys.executable,'scripts/check_array_events.py','--build-dir',ROOT/f'.build/array_{stages}_8x8','--pipelined-read','--timeout',7200)
        call(sys.executable,'scripts/check_array_cycles.py','--build-dir',ROOT/f'.build/array_{stages}_8x8','--events')
        call(sys.executable, 'scripts/check_auxiliary.py', '--stages', stages, '--dram')
        call(sys.executable,'scripts/check_double_buffer.py','--build-dir',ROOT/f'.build/system_{stages}_2x2',
             '--dram','--out',ROOT/f'artifacts/double_buffer_regression_{stages}')
        call(sys.executable,'scripts/check_descriptor.py','--stages',stages,'--dram')
        call(sys.executable,'scripts/check_native_prefetch.py','--build-dir',ROOT/f'.build/gemm_{stages}_2x2',
             '--dram','--out',ROOT/f'artifacts/native_prefetch_regression_{stages}')
        call(sys.executable,'scripts/check_grouped.py','--stages',stages,'--dram')
        call(sys.executable,'scripts/check_epilogue.py','--build-dir',ROOT/f'.build/gemm_{stages}_2x2',
             '--dram','--out',ROOT/f'artifacts/epilogue_regression_{stages}')
        call(sys.executable,'scripts/check_encoded_descriptor.py','--build-dir',ROOT/f'.build/gemm_{stages}_2x2',
             '--out',ROOT/f'artifacts/encoded_descriptor_regression_{stages}')
        call(sys.executable,'scripts/check_faults.py','--stages',stages,'--dram')
        call(sys.executable,'scripts/check_reset.py','--stages',stages,'--dram')
        call(sys.executable,'scripts/check_system_cdc.py','--build-dir',ROOT/f'.build/gemm_{stages}_2x2')
        call(sys.executable,'scripts/check_scratch.py','--full','--build-dir',ROOT/f'.build/gemm_{stages}_2x2',
             '--out',ROOT/f'artifacts/scratch_regression_{stages}')
        for fmt in ('I4', 'BF16', 'F64'):
            call(sys.executable, 'scripts/run_gemm.py', '--format', fmt, '--stages', stages,
                 '--core-ps', CORE_PS, '--memory-ps', MEMORY_PS, '--dram')
        for fmt in ('I4', 'I8', 'F4', 'F8'):
            call(sys.executable, 'scripts/run_quantized.py', '--format', fmt, '--stages', stages)
    call(sys.executable,'scripts/check_equivalence.py','--stages',2)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stages', nargs='+', type=int, choices=[2], default=[2])
    p.add_argument('--jobs', type=int, default=8)
    run(p.parse_args())
