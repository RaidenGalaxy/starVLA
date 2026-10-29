#!/usr/bin/env python3
"""Autonomous group-scaled W4A8/W8A8 GEMM; no host partial-sum schedule."""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys as _architecture_sys
_architecture_sys.path.insert(0,str(ROOT/'python'))
from simd_eval.architecture import CORE_PS, MEMORY_PS
sys.path.insert(0, str(ROOT / 'python'))
from runtime import validated_build
from simd_eval.quantization import quantize, load_codes
from simd_eval.reference import Format, Op, WIDTHS, encode, fp_operation, signed
from simd_eval.reporting import numerical_errors, utilization, workload_hash, write_metrics,dram_evidence,sequencer_cycle_evidence,dma_link_metrics,native_prefetch_evidence,epilogue_evidence
from simd_eval.epilogue import output_configuration,finish_matrix,packed_row,read_matrix

MODE_FORMATS={'W4A4':Format.I4,'W4A8':Format.I8,'W8A8':Format.I8,'FP4':Format.F4,'FP8':Format.F8}


def run(stages=2, rows=2, cols=2, dram=False, out=None, npz=None,
        mode=None, group_log2=5, core_ps=CORE_PS, build_dir=None, timeout=None,dram_config=None,memory_ps=MEMORY_PS,
        reference_backend='rational',quantized_npz=None,prefetch=True,
        output_format=None,output_scale='1',output_zero_point=0):
    if dram_config and not dram:raise ValueError('--dram-config requires --dram')
    if min(core_ps,memory_ps)<=0 or core_ps%2 or memory_ps%2:raise ValueError('clock periods must be positive even picoseconds')
    if reference_backend not in ('rational','cpu'):raise ValueError('unknown reference backend')
    reference_evidence=dict(reference_backend='exact_rational');reference_seconds=0.0
    config_path=Path(dram_config).resolve()if dram_config else ROOT/'configs/ddr4.ini'
    dest = Path(build_dir)if build_dir else ROOT / f'.build/gemm_{stages}_{rows}x{cols}'
    guard, caps = validated_build(dest)
    output=output_configuration(caps,output_format,output_scale,output_zero_point)
    output_width=WIDTHS[Format[output_format]]if output else 32
    if(caps['target'],caps['stages'],caps['rows'],caps['cols'])!=('gemm',stages,rows,cols):
        raise ValueError('build scope differs from grouped check')
    if not caps.get('grouped_int8'):
        raise ValueError('rebuild GEMM with autonomous grouped INT8 support')
    if not 0 <= group_log2 <= 16:
        raise ValueError('group log2 must be in [0,16]')
    out = Path(out) if out else ROOT / f'artifacts/grouped_{stages}_{rows}x{cols}_{"dram" if dram else "ideal"}'
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(817)
    original = None
    encoded = None
    framework_reference = None
    if npz and quantized_npz:raise ValueError('select original tensors or an encoded checkpoint')
    if quantized_npz:
        import numpy as np
        with np.load(quantized_npz,allow_pickle=False)as checkpoint:
            encoded_format=Format[str(checkpoint['format'].item())]
        symmetric_modes={Format.I4:'W4A4',Format.I8:'W8A8',Format.F4:'FP4',Format.F8:'FP8'}
        if encoded_format not in symmetric_modes:raise ValueError('encoded descriptor inputs require I4/I8/F4/F8')
        inferred=symmetric_modes[encoded_format]
        if mode is not None and mode!=inferred:raise ValueError('operand mode differs from encoded checkpoint format')
        mode=inferred
        a,b,qa,qb,sa,sb,group_size=load_codes(quantized_npz,encoded_format)
        if group_size>65536 or group_size&(group_size-1):
            raise ValueError('native descriptor group size must be a power of two in [1,65536]')
        group_log2=group_size.bit_length()-1
        original=(a,b);encoded=(qa,qb,sa,sb)
    mode=mode or 'W4A8'
    modes=('W4A4','W4A8','W8A8')if caps.get('grouped_int4')else('W4A8','W8A8')
    if caps.get('grouped_fp4_fp8'):modes+=('FP4','FP8')
    cases = [(mode, m, n, k, g) for mode in modes
             for m, n, k, g in [(1, 1, 1, 0), (3, 19, 17, 0), (3, 19, 17, 2),
                                (3, 19, 17, 4), (3, 7, 65, 5), (1, cols*(64//WIDTHS[MODE_FORMATS[mode]])*17+3, 3, 1)]]
    if npz:
        import numpy as np
        data = np.load(npz, allow_pickle=False)
        a, b = data['a'], data['b']
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or min(*a.shape, *b.shape) < 1:
            raise ValueError('expected nonempty A[M,K] and B[K,N]')
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('finite source tensors required')
        original = ([[Fraction(float(v)) for v in row] for row in a],
                    [[Fraction(float(v)) for v in row] for row in b])
        if 'reference_output'in data:
            reference=data['reference_output']
            if reference.shape!=(a.shape[0],b.shape[1])or not np.isfinite(reference).all():
                raise ValueError('invalid framework reference output')
            framework_reference=[[Fraction(float(v))for v in row]for row in reference]
        cases = [(mode, a.shape[0], b.shape[1], a.shape[1], group_log2)]
    elif encoded is not None:
        cases=[(mode,len(a),len(b[0]),len(a[0]),group_log2)]
    commands, expected, jobs = [], {}, {}
    if caps.get('autonomous_prefetch'):commands.append(f'V {int(prefetch)}\n')
    if output:commands.append(f'O 1 {int(Format[output_format])} {output["output_scale_bits"]:x} {output_zero_point}\n')
    actual_layout = []

    def tensor(base, stride, matrix, bits):
        data = bytearray(len(matrix)*stride)
        for i, values in enumerate(matrix):
            value = sum((v & ((1 << bits)-1)) << (j*bits) for j, v in enumerate(values))
            size = (len(values)*bits+7)//8
            data[i*stride:i*stride+size] = value.to_bytes(size, 'little')
        for offset in range(0, len(data), 16):
            commands.append(f'L {base+offset} {int.from_bytes(data[offset:offset+16], "little"):032x}\n')

    for index, (operand_mode, m, n, k, g) in enumerate(cases):
        if not 1 <= k <= 65536:
            raise ValueError('K outside hardware limit')
        group_size = 1 << g
        groups = (k+group_size-1)//group_size
        a, b = original if original else (
            [[Fraction(rng.randrange(-100, 101), 13) for _ in range(k)] for _ in range(m)],
            [[Fraction(rng.randrange(-100, 101), 17) for _ in range(n)] for _ in range(k)])
        fmt=MODE_FORMATS[operand_mode];activation_bits=WIDTHS[fmt]
        floating=fmt>=Format.F4
        if fmt==Format.I4 and not caps.get('grouped_int4'):raise ValueError('build lacks autonomous grouped W4A4')
        if floating and not caps.get('grouped_fp4_fp8'):raise ValueError('build lacks autonomous grouped FP4/FP8')
        qa, qb, sa, sb = encoded if encoded is not None else quantize(a, b, fmt, group_size)
        weight4 = operand_mode == 'W4A8'
        if weight4:
            _, qb, _, sb = quantize(a, b, Format.I4, group_size)
        reference_start=time.perf_counter()
        cpu_partials=[]
        if floating and reference_backend=='cpu':
            import numpy as np
            from simd_eval.cpu_reference import gemm as cpu_gemm
            encoded_a=np.asarray(qa,dtype=np.uint64);encoded_b=np.asarray(qb,dtype=np.uint64)
            for group in range(groups):
                lo,hi=group*group_size,min(k,(group+1)*group_size)
                values,status,evidence=cpu_gemm(encoded_a[:,lo:hi],encoded_b[lo:hi,:],fmt)
                cpu_partials.append((values,status))
            reference_evidence={**evidence,'reference_backend':'host_cpu_group_fma_rational_scale',
                'reference_group_semantics':'sequential K FP32 FMA within each group; exact rational FP32 scale multiply and group accumulation'}
        bits = 4 if weight4 else activation_bits
        result = [[0]*n for _ in range(m)]
        flags = 0
        for i in range(m):
            for j in range(n):
                for group in range(groups):
                    lo, hi = group*group_size, min(k, (group+1)*group_size)
                    if floating:
                        if cpu_partials:
                            values,status=cpu_partials[group];converted=values[i][j];ff=status[i][j]
                        else:
                            converted=0;ff=0
                            for t in range(lo,hi):
                                converted,pf=fp_operation(qa[i][t],qb[t][j],converted,fmt,Op.MAC);ff|=pf
                    else:
                        partial = sum(signed(qa[i][t], activation_bits)*signed(qb[t][j], bits) for t in range(lo, hi))
                        converted, ff = encode(Fraction(partial), Format.F32)
                    gamma, gf = fp_operation(sa[i][group], sb[group][j], 0, Format.F32, Op.MUL)
                    result[i][j], sf = fp_operation(converted, gamma, result[i][j], Format.F32, Op.MAC)
                    flags |= ff | gf | sf
        reference_seconds+=time.perf_counter()-reference_start
        result,output_flags,output_clips=finish_matrix(result,Format.F32,output)
        flags|=output_flags
        base = index*0x100000
        aa, bb, cc = 0x10000000+base, 0x30000000+base, 0x80000003+base+index%64
        sa_address, sb_address = 0x50000000+base, 0x60000000+base
        astride, bstride = ((k*activation_bits+63)//64)*8+8, ((n*bits+63)//64)*8+16
        cstride = (n*output_width+7)//8+13
        sa_stride, sb_stride = ((m*4+7)//8)*8+8, ((n*4+7)//8)*8+8
        tensor(aa, astride, qa, activation_bits)
        tensor(bb, bstride, qb, bits)
        tensor(sa_address, sa_stride, [list(values) for values in zip(*sa)], 32)
        tensor(sb_address, sb_stride, sb, 32)
        for i in range(m):
            for offset, value in enumerate(packed_row(result[i],output_width)):
                expected[cc+i*cstride+offset] = value
        commands.append(f'G {int(fmt)} {m} {n} {k} {aa} {bb} {cc} {astride} {bstride} {cstride} {index} '
                        f'{int(weight4)} 0 {sb_address} {sb_stride} {g} 1 {sa_address} {sa_stride}\n')
        auxiliary_lanes=caps.get('group_auxiliary_lanes',1)
        if auxiliary_lanes not in(1,2):raise ValueError('unsupported grouped auxiliary lane count')
        pairs=m*((n+auxiliary_lanes-1)//auxiliary_lanes)
        jobs[index] = dict(format=fmt.name,operand_mode=operand_mode, m=m, n=n, k=k, group_size=group_size,
                           groups=groups, flags=flags, output_vectors=pairs,
                           expected_output_saturations=output_clips,
                           expected_auxiliary_operations=(2 if floating else 3)*pairs*groups,
                           auxiliary_scalar_operations=(2 if floating else 3)*m*n*groups,
                           logical_input_bytes=m*((k*activation_bits+7)//8)+k*((n*bits+7)//8),
                           logical_scale_bytes=4*groups*(m+n),logical_output_bytes=m*((n*output_width+7)//8),
                           workload_sha256=workload_hash(a, b))
        actual_layout.append((cc, cstride))
    trace = out / 'commands.txt'
    trace.write_text(''.join(commands))
    with (out / 'simulation.log').open('w') as log:
        subprocess.run([str(dest/'sim'), str(trace), str(core_ps), str(memory_ps),
                        str(config_path) if dram else '-', str(out/'dram'), '207'],
                       stdout=log, stderr=subprocess.STDOUT, check=True,
                       timeout=timeout if timeout is not None else max(1800,120*rows*cols))
    written, done, summary, onchip, accepted_start = {}, set(), None, None, None
    for line in (out / 'simulation.log').read_text().splitlines():
        if line.startswith('W '):
            _, address, value = line.split()
            written[int(address)] = int(value)
        elif line.startswith('D '):
            values = list(map(int, line.split()[1:]))
            if len(values) != 13:
                raise AssertionError('missing completion counters')
            tag, flags, cycles, macs, overflow, memory_cycles, reads, ps = values[:8]
            if tag in done or tag not in jobs:
                raise AssertionError('duplicate/unexpected completion')
            done.add(tag)
            job = jobs[tag]
            if flags != job['flags'] or macs != job['m']*job['n']*job['k'] or overflow:
                raise AssertionError(f'grouped arithmetic/counter mismatch: {values}, {job}')
            if values[11] != job['expected_auxiliary_operations']:
                raise AssertionError('conversion, scale or aggregate operation was omitted')
            job.update(compute_cycles=cycles, macs=macs, memory_cycles=memory_cycles, read_bursts=reads,
                       auxiliary_operations=values[11], auxiliary_memory_cycles=values[12])
            job.update(utilization(*values[8:11], cycles, macs, rows*cols, 64//WIDTHS[Format[job['format']]]))
        elif line.startswith('S '):
            summary = list(map(int, line.split()[1:]))
        elif line.startswith('H '):
            onchip = list(map(int, line.split()[1:]))
        elif line.startswith('T '):
            if accepted_start is not None:raise AssertionError('duplicate first-command timestamp')
            _,first_tag,accepted_start=line.split();accepted_start=int(accepted_start)
    if written != expected:
        for address in sorted(set(written) | set(expected)):
            if written.get(address) != expected.get(address):
                raise AssertionError(f'grouped output {address:#x}: {written.get(address)} != {expected.get(address)}')
    if len(done) != len(jobs) or summary is None or onchip is None:
        raise AssertionError('missing completion/memory evidence')
    start, submitted, end, reads, writes, payload, rb, wb = summary
    traffic=dram_evidence((out/'simulation.log').read_text(),reads,writes,rb,wb,dram)
    if accepted_start is not None and start!=accepted_start:raise AssertionError('timer/command handshake mismatch')
    if sum(job['read_bursts'] for job in jobs.values()) != reads or payload != len(expected):
        raise AssertionError('gather/scatter accounting mismatch')
    expected_aggregate_writes = 8*sum(job['output_vectors']*job['groups'] for job in jobs.values())
    expected_aggregate_reads = 8*sum(job['output_vectors']*(job['groups']-1) for job in jobs.values())
    if len(onchip) != 4 or onchip[2:] != [expected_aggregate_reads, expected_aggregate_writes]:
        raise AssertionError('group aggregate state traffic mismatch')
    macs = sum(job['macs'] for job in jobs.values())
    metrics = dict(passed=True, execution_path='autonomous_rtl_grouped_gemm',input_schedule='descriptor_sequential_gather',
        test_scope='workload' if original else 'full', stages=stages, rows=rows, cols=cols,
        sram_read_latency=caps.get('sram_read_latency',1),
        descriptors=len(jobs), formats=sorted({job['format']for job in jobs.values()}), operand_modes=sorted({job['operand_mode'] for job in jobs.values()}),
        exact_output_match=True, verified_output_bytes=len(written), macs=macs,
        compute_cycles=sum(job['compute_cycles'] for job in jobs.values()), elapsed_ps=end-start,
        elapsed_time_origin='first_descriptor_acceptance'if accepted_start is not None else'simulation_after_reset',
        submitted_ps=submitted-start, core_frequency_hz=1e12/core_ps, memory_frequency_hz=1e12/memory_ps,
        dequant_operations=sum(job['auxiliary_operations'] for job in jobs.values()),
        group_auxiliary_lanes=caps.get('group_auxiliary_lanes',1),
        auxiliary_scalar_operations=sum(job['auxiliary_scalar_operations']for job in jobs.values()),
        dequant_memory_cycles=sum(job['auxiliary_memory_cycles'] for job in jobs.values()),
        dma_read_requests=reads, dma_write_requests=writes, output_payload_bytes=payload,
        dma_transfer_bytes=64*(reads+writes), dram_read_bursts=rb, dram_write_bursts=wb,
        dram_physical_bytes=traffic['dram_physical_bytes'], dram=dram, dma_backend=caps.get('dma_backend', 'rtl'),
        sram_read_bytes=onchip[0], sram_write_bytes=onchip[1],
        aggregate_read_bytes=onchip[2], aggregate_write_bytes=onchip[3], onchip_moved_bytes=sum(onchip),
        timing_validated=False, time_kind='configured_clock_projection',
        rtl_sha256=caps['rtl_sha256'], simulator_sha256=caps['simulator_sha256'],
        testbench_sha256=caps['testbench_sha256'], dram_library_sha256=caps['dram_library_sha256'],
        dram_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest() if dram else None,
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
    metrics.update(traffic)
    metrics.update(gops=2*macs/((end-start)*1e-12)/1e9,
                   gmac_s=macs/((end-start)*1e-12)/1e9,
                   dma_bandwidth_GB_s=metrics['dma_transfer_bytes']/((end-start)*1e-12)/1e9,
                   dram_bandwidth_GB_s=traffic['dram_physical_bytes']/((end-start)*1e-12)/1e9 if dram else None)
    for field in('logical_input_bytes','logical_scale_bytes','logical_output_bytes'):
        metrics[field]=sum(job[field]for job in jobs.values())
    metrics['useful_transfer_bytes']=sum(metrics[field]for field in
        ('logical_input_bytes','logical_scale_bytes','logical_output_bytes'))
    metrics['useful_bandwidth_GB_s']=metrics['useful_transfer_bytes']/((end-start)*1e-12)/1e9
    if original:
        a, b = original
        exact = [[sum(a[i][t]*b[t][j] for t in range(k)) for j in range(n)] for i in range(m)]
        cc, stride = actual_layout[0]
        actual = read_matrix(written,cc,stride,m,n,output_width)
        metrics.update(operand_mode=mode, format=jobs[0]['format'], m=m, n=n, k=k, group_size=1 << group_log2,
                       workload_sha256=workload_hash(a, b),
                       source_npz_sha256=hashlib.sha256(Path(quantized_npz or npz).read_bytes()).hexdigest())
        if encoded is not None:
            metrics.update(quantized_codes_imported=True,quantized_checkpoint_sha256=metrics['source_npz_sha256'],
                           quantization_source='external_encoded_checkpoint_without_requantization')
        error_format=Format[output_format]if output else Format.F32
        error_options=dict(scale=Fraction(output['output_scale_exact']),zero_point=output_zero_point)if output else{}
        metrics.update(numerical_errors(actual, exact,error_format,**error_options))
        if framework_reference is not None:
            metrics.update({'framework_'+key:value for key,value in numerical_errors(actual,framework_reference,error_format,**error_options).items()})
    if output:metrics.update(output)
    metrics.update(epilogue_evidence((out/'simulation.log').read_text(),caps,jobs,bool(output)))
    metrics.update(sequencer_cycle_evidence((out/'simulation.log').read_text(),caps,jobs))
    metrics.update(native_prefetch_evidence((out/'simulation.log').read_text(),caps,jobs))
    metrics['native_prefetch_requested']=bool(prefetch and caps.get('autonomous_prefetch'))
    if metrics['native_prefetch_requested']:metrics['input_schedule']='descriptor_wave_fifo_sram_prefetch'
    metrics.update(dma_link_metrics(metrics,reads,writes))
    metrics.update(reference_evidence);metrics['reference_host_seconds']=reference_seconds
    metrics.update({key:caps[key] for key in ('process','architecture_sha256','dependency_lock_sha256','sram_macro','storage_implementation','sram_physical_certified')})
    write_metrics(out, metrics)
    (out/'jobs.json').write_text(json.dumps(jobs, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stages', type=int, default=2)
    p.add_argument('--rows', type=int, default=2)
    p.add_argument('--cols', type=int, help='defaults to --rows')
    p.add_argument('--dram', action='store_true')
    p.add_argument('--out')
    p.add_argument('--npz')
    p.add_argument('--output-format',choices=[fmt.name for fmt in Format])
    p.add_argument('--output-scale',default='1',help='multiply by this FP32 scale before output RNE conversion')
    p.add_argument('--output-zero-point',type=int,default=0)
    p.add_argument('--quantized-npz',help='encoded checkpoint qa/qb and FP32 scale bits, sent directly to native RTL descriptors')
    p.add_argument('--mode', choices=list(MODE_FORMATS),help='default W4A8; inferred for encoded symmetric checkpoints')
    p.add_argument('--group-log2', type=int, default=5)
    p.add_argument('--core-ps', type=int, default=CORE_PS)
    p.add_argument('--memory-ps',type=int,default=MEMORY_PS)
    p.add_argument('--build-dir')
    p.add_argument('--no-prefetch',dest='prefetch',action='store_false',help='serialize next-tile loading on the same native hardware')
    p.add_argument('--dram-config')
    p.add_argument('--timeout',type=int,help='host runtime limit in seconds; default scales with PE count')
    p.add_argument('--reference-backend',choices=['rational','cpu'],default='rational',
                   help='optional sequential host FMA within FP4/FP8 groups; integer partials and group scales remain exact rational')
    args=p.parse_args()
    if args.cols is None:args.cols=args.rows
    run(**vars(args))
