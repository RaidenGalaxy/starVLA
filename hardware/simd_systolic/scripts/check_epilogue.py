#!/usr/bin/env python3
"""Check device output scaling/casting, packing and chained descriptors.

Only inputs, scale bits and descriptors enter simulated memory. Expected
outputs are computed separately with exact rational arithmetic.
"""
import argparse
import hashlib
import json
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys as _architecture_sys
_architecture_sys.path.insert(0,str(ROOT/'python'))
from simd_eval.architecture import CORE_PS, MEMORY_PS
sys.path.insert(0, str(ROOT / 'python'))
from runtime import validated_build
from simd_eval.epilogue import apply
from simd_eval.reference import Format, Op, WIDTHS, ENCODINGS, gemm, to_bits, signed, encode, fp_operation
from simd_eval.reporting import (write_metrics, dram_evidence, sequencer_cycle_evidence,
                                 native_prefetch_evidence, dma_link_metrics)


def check(build, out, dram=False, core_ps=CORE_PS, memory_ps=MEMORY_PS, timeout=7200, smoke=False,dram_config=None):
    guard, caps = validated_build(build)
    if caps['target'] != 'gemm' or not caps.get('output_epilogue'):
        raise ValueError('an output-epilogue GEMM build is required')
    if min(core_ps, memory_ps) <= 0 or core_ps % 2 or memory_ps % 2:
        raise ValueError('clock periods must be positive even picoseconds')
    if dram_config and not dram:raise ValueError('--dram-config requires --dram')
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(6821)
    commands, expected, jobs = [], {}, {}
    if caps.get('autonomous_prefetch'):
        commands.append('V 1\n')

    def tensor(base, stride, matrix, bits):
        data = bytearray(len(matrix)*stride)
        for row, values in enumerate(matrix):
            raw = sum((value & ((1 << bits)-1)) << (col*bits)
                      for col, value in enumerate(values))
            size = (len(values)*bits+7)//8
            data[row*stride:row*stride+size] = raw.to_bytes(size, 'little')
        for offset in range(0, len(data), 16):
            commands.append(f'L {base+offset} {int.from_bytes(data[offset:offset+16], "little"):032x}\n')

    previous = None

    def job(source, destination, m, n, k, scale, zero=0, grouped=False, weight4=False, dependent=False, chain_output=False, special=False):
        nonlocal previous
        tag = len(jobs)
        width, output_width = WIDTHS[source], WIDTHS[destination]
        weight_width = 4 if weight4 else width

        def value():
            number = rng.randrange(-7, 8)
            return number & ((1 << width)-1) if source < Format.F4 else to_bits(Fraction(number, 2), source)

        a = [[value() for _ in range(k)] for _ in range(m)]
        b = [[value() for _ in range(n)] for _ in range(k)]
        if weight4:
            b = [[rng.randrange(16) for _ in range(n)] for _ in range(k)]
        if dependent:
            if previous is None or previous['format'] != source or previous['shape'] != (m, k):
                raise ValueError('invalid dependent descriptor')
            a = previous['codes']
        # The INT64 zero-point -1 case must reach MAX64 rather than MAX64-1.
        if source < Format.F4 and not dependent:
            a[0][0] = 1 << (width-1)
            b[0][0] = (-1) & ((1 << weight_width)-1)
        if special:
            exponent, mantissa = ENCODINGS[source]
            a[0][0] = (((1 << exponent)-2) << mantissa) | ((1 << mantissa)-1)
            a[1][0] = 1
            a[2][0] = (((1 << exponent)-1) << mantissa) | 1
            b[0][0] = to_bits(Fraction(2), source)
            b[0][1] = 1
        if grouped:
            group_log2 = 2
            group_size = 1 << group_log2
            groups = (k+group_size-1)//group_size
            sa = [[to_bits(Fraction((i+g) % 4, 3), Format.F32)
                   for g in range(groups)] for i in range(m)]
            sb = [[to_bits(Fraction((j+2*g) % 7-3, 5), Format.F32)
                   for j in range(n)] for g in range(groups)]
            native = [[0]*n for _ in range(m)]
            flags = 0
            for i in range(m):
                for j in range(n):
                    for group in range(groups):
                        lo, hi = group*group_size, min(k, (group+1)*group_size)
                        if source < Format.F4:
                            total = sum(signed(a[i][t], width)*signed(b[t][j], weight_width)
                                        for t in range(lo, hi))
                            partial, ff = encode(Fraction(total), Format.F32)
                        else:
                            partial, ff = 0, 0
                            for t in range(lo, hi):
                                partial, pf = fp_operation(a[i][t], b[t][j], partial, source, Op.MAC)
                                ff |= pf
                        gamma, gf = fp_operation(sa[i][group], sb[group][j], 0, Format.F32, Op.MUL)
                        native[i][j], sf = fp_operation(partial, gamma, native[i][j], Format.F32, Op.MAC)
                        flags |= ff | gf | sf
            epilogue_source = Format.F32
        else:
            native, statuses = gemm(a, b, source, return_flags=True)
            flags = 0
            for row in statuses:
                for status in row:
                    flags |= status
            epilogue_source = source
        result, clips = [], 0
        for row in native:
            codes = []
            for value in row:
                code, status, clipped = apply(value, epilogue_source, destination, scale, zero)
                codes.append(code)
                flags |= status
                clips += int(clipped)
            result.append(codes)
        aa, bb, cc = 0x10000000+tag*0x100000, 0x30000000+tag*0x100000, 0x80000003+tag*0x100000+tag % 61
        astride = ((k*width+63)//64)*8+8
        bstride = ((n*weight_width+63)//64)*8+16
        cstride = (n*output_width+7)//8+13
        if chain_output:
            cstride = (cstride+7)//8*8
            cc = (cc+7)//8*8
        if dependent:
            aa, astride = previous['address'], previous['stride']
        else:
            tensor(aa, astride, a, width)
        tensor(bb, bstride, b, weight_width)
        scale_a, scale_b = 0x50000000+tag*0x100000, 0x60000000+tag*0x100000
        scale_a_stride, scale_b_stride = ((m*32+63)//64)*8+8, ((n*32+63)//64)*8+8
        if grouped:
            tensor(scale_a, scale_a_stride, list(map(list, zip(*sa))), 32)
            tensor(scale_b, scale_b_stride, sb, 32)
        size = (n*output_width+7)//8
        for row, values in enumerate(result):
            raw = sum(value << (column*output_width) for column, value in enumerate(values))
            for offset, value in enumerate(raw.to_bytes(size, 'little')):
                expected[cc+row*cstride+offset] = value
        commands.append(f'O 1 {int(destination)} {scale:x} {zero}\n')
        commands.append(f'G {int(source)} {m} {n} {k} {aa} {bb} {cc} {astride} {bstride} {cstride} {tag} '
                        f'{int(weight4)} 0 {scale_b} {scale_b_stride} 2 {int(grouped)} {scale_a} {scale_a_stride}\n')
        jobs[tag] = dict(fmt=source.name, output_format=destination.name, m=m, n=n, k=k,
                         flags=flags, saturations=clips, grouped=grouped, weight4=weight4,
                         dependent=dependent, scale_bits=scale, zero_point=zero,
                         exceptional_input=special,
                         logical_output_bytes=m*size)
        previous = dict(codes=result, address=cc, stride=cstride, format=destination, shape=(m, n))

    destinations = [Format.I4, Format.F4, Format.F32,Format.F64] if smoke else list(Format)
    for source in Format:
        for destination in destinations:
            job(source, destination, 1, 1, 1, 0x3f800000, -1 if destination < Format.F4 else 0)
            if not smoke:
                job(source, destination, 3, 7, 9, to_bits(Fraction(3, 8), Format.F32),
                    3 if destination < Format.F4 else 0,
                    special=source > Format.F4)
    # One-lane sources join nibbles across PE columns, contexts and N tiles.
    for source in (Format.I64, Format.F64):
        for destination in (Format.I4, Format.F4):
            job(source, destination, 3, caps['cols']*17+1, 9, to_bits(Fraction(1, 32), Format.F32))
    for source, weight4 in ((Format.I4, False), (Format.I8, False), (Format.I8, True),
                             (Format.F4, False), (Format.F8, False)):
        for destination in destinations:
            job(source, destination, 3, 19, 9, to_bits(Fraction(3, 4), Format.F32),
                -2 if destination < Format.F4 else 0, grouped=True, weight4=weight4)
    # Output of descriptor one is descriptor two's A; it is never host loaded.
    for middle in (Format.I4, Format.I8, Format.F4, Format.F16, Format.F32, Format.F64):
        job(Format.F32, middle, 3, 3, 3, 0x3f800000, chain_output=True)
        job(middle, Format.F32, 3, 7, 3, 0x3f800000, dependent=True)
    trace = out/'commands.txt'
    trace.write_text(''.join(commands))
    config = Path(dram_config).resolve()if dram_config else ROOT/'configs/ddr4.ini'
    with (out/'simulation.log').open('w') as log:
        subprocess.run([str(build/'sim'), str(trace), str(core_ps), str(memory_ps),
                        str(config) if dram else '-', str(out/'dram'), '207'],
                       check=True, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    stdout = (out/'simulation.log').read_text()
    written, done, converted = {}, set(), set()
    summary = None
    for line in stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'W':
            _, address, value = parts
            address = int(address)
            if address in written:
                raise AssertionError(f'duplicate output byte {address:#x}')
            written[address] = int(value)
        elif parts[0] == 'D':
            values = list(map(int, parts[1:]))
            tag, flags, cycles, macs, overflow, memory_cycles, reads = values[:7]
            if tag in done or tag not in jobs:
                raise AssertionError('duplicate/unexpected completion')
            done.add(tag)
            job_data = jobs[tag]
            if flags != job_data['flags'] or macs != job_data['m']*job_data['n']*job_data['k'] or overflow:
                raise AssertionError(f'arithmetic/count mismatch {values}, expected {job_data}')
            job_data.update(compute_cycles=cycles, memory_cycles=memory_cycles, read_bursts=reads)
        elif parts[0] == 'Y':
            tag, elements, auxiliary, cycles, clips = map(int, parts[1:])
            job_data = jobs[tag]
            if tag in converted or elements != job_data['m']*job_data['n'] or clips != job_data['saturations']:
                raise AssertionError(f'epilogue counts {parts}, expected {job_data}')
            if not 0 < auxiliary <= elements*2 or cycles < auxiliary:
                raise AssertionError('output conversion operations/cycles omitted')
            converted.add(tag)
            job_data.update(epilogue_elements=elements, epilogue_auxiliary_vectors=auxiliary,
                            epilogue_memory_cycles=cycles)
        elif parts[0] == 'S':
            summary = list(map(int, parts[1:]))
    if written != expected:
        for address in sorted(set(written) | set(expected)):
            if written.get(address) != expected.get(address):
                raise AssertionError(f'output byte {address:#x}: {written.get(address)} != {expected.get(address)}')
    if done != set(jobs) or converted != set(jobs) or summary is None:
        raise AssertionError('missing command/output conversion completion')
    start, submitted, end, reads, writes, payload, read_bursts, write_bursts = summary
    if payload != len(expected) or sum(j['read_bursts'] for j in jobs.values()) != reads:
        raise AssertionError('output/input byte accounting mismatch')
    metrics = dict(passed=True, scope='descriptor_output_epilogue', stages=caps['stages'],
                   rows=caps['rows'], cols=caps['cols'], all_input_formats=True,
                   all_output_formats=not smoke, descriptors=len(jobs),
                   grouped_descriptors=sum(j['grouped'] for j in jobs.values()),
                   dependent_descriptors=sum(j['dependent'] for j in jobs.values()),
                   arithmetic='native_accumulator_to_FP32_then_FP32_scale_then_RNE_cast',
                   integer_zero_point_before_saturation=True, packed_int4_fp4=True,
                   packed_odd_row_padding_zero=True, exact_output_match=True,
                   output_payload_bytes=payload, dma_read_requests=reads, dma_write_requests=writes,
                   verified_output_bytes=len(expected), elapsed_ps=end-start,
                   exceptional_descriptors=sum(j['exceptional_input'] for j in jobs.values()),
                   epilogue_elements=sum(j['epilogue_elements'] for j in jobs.values()),
                   epilogue_auxiliary_vectors=sum(j['epilogue_auxiliary_vectors'] for j in jobs.values()),
                   epilogue_memory_cycles=sum(j['epilogue_memory_cycles'] for j in jobs.values()),
                   epilogue_saturations=sum(j['saturations'] for j in jobs.values()),
                   submitted_ps=submitted-start, macs=sum(j['m']*j['n']*j['k'] for j in jobs.values()),
                   core_frequency_hz=1e12/core_ps, memory_frequency_hz=1e12/memory_ps,
                   time_kind='configured_clock_projection', timing_validated=False, dram=dram,
                   dram_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest()if dram else None,
                   rtl_sha256=caps['rtl_sha256'], simulator_sha256=caps['simulator_sha256'],
                   testbench_sha256=caps['testbench_sha256'], dram_library_sha256=caps['dram_library_sha256'],
                   trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                   output_sha256=hashlib.sha256(bytes(written[a] for a in sorted(written))).hexdigest())
    metrics.update(dram_evidence(stdout, reads, writes, read_bursts, write_bursts, dram))
    metrics.update(sequencer_cycle_evidence(stdout, caps, jobs))
    metrics.update(native_prefetch_evidence(stdout, caps, jobs))
    metrics.update(dma_link_metrics(metrics, reads, writes))
    metrics.update({key:caps[key] for key in ('process','architecture_sha256','dependency_lock_sha256','sram_macro','storage_implementation','sram_physical_certified')})
    write_metrics(out, metrics)
    (out/'jobs.json').write_text(json.dumps(jobs, indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dram', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dram-config')
    parser.add_argument('--core-ps', type=int, default=CORE_PS)
    parser.add_argument('--memory-ps', type=int, default=MEMORY_PS)
    parser.add_argument('--timeout', type=int, default=7200)
    args = parser.parse_args()
    check(args.build_dir.resolve(), args.out.resolve(), args.dram,
          args.core_ps, args.memory_ps, args.timeout, args.smoke,args.dram_config)
