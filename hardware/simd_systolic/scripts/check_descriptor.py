#!/usr/bin/env python3
"""No host tiling: supply row-major tensors and one RTL GEMM descriptor per job."""
import argparse,hashlib,json,random,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
import sys as _architecture_sys
_architecture_sys.path.insert(0,str(ROOT/'python'))
from simd_eval.architecture import CORE_PS, MEMORY_PS
from simd_eval.reference import Format,Op,WIDTHS,ACC_WIDTHS,ENCODINGS,gemm,to_bits,fp_operation,narrow32,signed
from simd_eval.hardware_contract import hardware_contract
from simd_eval.validation import require_output_bytes
from simd_eval.reporting import write_metrics,utilization,workload_hash,numerical_errors,dram_evidence,sequencer_cycle_evidence,dma_link_metrics,native_prefetch_evidence,epilogue_evidence
from simd_eval.epilogue import output_configuration,finish_matrix,packed_row,read_matrix
from simd_eval.quantization import quantize
from runtime import validated_build


def check(stages=2,rows=2,cols=2,dram=False,out=None,core_ps=CORE_PS,smoke=False,npz=None,format='BF16',mode='symmetric',group_log2=5,build_dir=None,timeout=None,dram_config=None,memory_ps=MEMORY_PS,reference_backend='rational',prefetch=True,output_format=None,output_scale='1',output_zero_point=0):
    if dram_config and not dram:raise ValueError('--dram-config requires --dram')
    if min(core_ps,memory_ps)<=0 or core_ps%2 or memory_ps%2:raise ValueError('clock periods must be positive even picoseconds')
    if reference_backend not in('rational','cpu')or(reference_backend=='cpu'and not npz):
        raise ValueError('CPU reference requires an FP16/BF16 NPZ workload')
    reference_evidence=dict(reference_backend='exact_rational');reference_seconds=0.0
    config_path=Path(dram_config).resolve()if dram_config else ROOT/'configs/ddr4.ini'
    dest=Path(build_dir)if build_dir else ROOT/f'.build/gemm_{stages}_{rows}x{cols}';guard,caps=validated_build(dest)
    output=output_configuration(caps,output_format,output_scale,output_zero_point)
    if output and not npz:raise ValueError('workload output options require --npz; use check_epilogue.py for the full conversion regression')
    if(caps['target'],caps['stages'],caps['rows'],caps['cols'])!=('gemm',stages,rows,cols):raise ValueError('build scope differs from descriptor check')
    out=Path(out)if out else ROOT/f'artifacts/descriptor_{stages}_{rows}x{cols}_{"dram"if dram else"ideal"}{"_smoke"if smoke else""}'
    out.mkdir(parents=True,exist_ok=True)
    rng=random.Random(1701);commands=[];expected={};jobs={};total_macs=0
    if caps.get('autonomous_prefetch'):commands.append(f'V {int(prefetch)}\n')
    if output:commands.append(f'O 1 {int(Format[output_format])} {output["output_scale_bits"]:x} {output_zero_point}\n')
    cases=[(fmt,m,n,k)for fmt in Format for m,n,k in [(1,1,1),(7,19,65)]]
    cases += [(Format.I4,3,cols*16*17+3,9),(Format.F64,3,cols*17+1,17)]
    if smoke:cases=[(fmt,1,1,1)for fmt in Format]+[(Format.I4,3,39,9),(Format.F64,3,3,3)]
    initial_indices=set(range(len(cases),len(cases)+len(Format)))
    cases += [(fmt,3,7,9)for fmt in Format]
    mixed_index=len(cases);cases.append((Format.I8,3,39,9))
    dequant_indices=set(range(len(cases),len(cases)+2))if caps.get('w4a16_dequant')else set()
    if dequant_indices:cases += [(Format.F16,3,19,17),(Format.BF16,3,19,17)]
    chain_begin=len(cases)
    cases += [(Format.F32,3,3,3),(Format.F32,3,3,3)]
    original=None;framework_reference=None
    if npz:
        import numpy as np
        data=np.load(npz,allow_pickle=False);aa=data['a'];bb=data['b'];fmt=Format[format]
        if aa.ndim!=2 or bb.ndim!=2 or aa.shape[1]!=bb.shape[0] or min(*aa.shape,*bb.shape)<1:
            raise ValueError('expected finite A[M,K], B[K,N]')
        if not np.isfinite(aa).all()or not np.isfinite(bb).all():raise ValueError('finite workload required')
        if fmt not in(Format.F16,Format.BF16):raise ValueError('workload mode uses FP16/BF16 activations')
        if not 0<=group_log2<=16:raise ValueError('group log2 must be in [0,16]')
        if mode=='W4A16'and not caps.get('w4a16_dequant'):raise ValueError('build lacks W4A16 dequantization')
        original=([[Fraction(float(v))for v in row]for row in aa],[[Fraction(float(v))for v in row]for row in bb])
        m,k=aa.shape;n=bb.shape[1]
        if 'reference_output'in data:
            reference=data['reference_output']
            if reference.shape!=(m,n)or not np.isfinite(reference).all():raise ValueError('invalid framework reference output')
            framework_reference=[[Fraction(float(v))for v in row]for row in reference]
        if k>65536:raise ValueError('K exceeds hardware limit')
        cases=[(fmt,m,n,k)];initial_indices=set();mixed_index=-1;chain_begin=-2
        dequant_indices={0}if mode=='W4A16'else set()
    previous=None
    for index,(fmt,m,n,k)in enumerate(cases):
        width=WIDTHS[fmt];aw=ACC_WIDTHS[fmt];ab=aw//8
        accumulate=index in initial_indices
        floating_mixed=index in dequant_indices
        weight4=index==mixed_index or floating_mixed;weight_width=4 if weight4 else width
        def value():
            x=rng.randrange(-3,4)
            return x&((1<<width)-1)if fmt<Format.F4 else to_bits(Fraction(x,2),fmt)
        a=[[value()for _ in range(k)]for _ in range(m)]
        b=[[value()for _ in range(n)]for _ in range(k)]
        if original:
            a=[[to_bits(v,fmt)for v in row]for row in original[0]]
            b=[[to_bits(v,fmt)for v in row]for row in original[1]]
        dependent=index==chain_begin+1
        if dependent:a=previous['reference']
        # Exercise full-width products and carry propagation, not only small integers.
        if fmt<Format.F4 and not original:
            a[0][0]=1<<(width-1);b[0][0]=1<<(width-1)
            if weight4:b[0][0]=(-8)&255
        elif not original and fmt!=Format.F4 and m>2 and n>2 and not dependent and not floating_mixed:
            e,f=ENCODINGS[fmt]
            a[0][0]=(((1<<e)-2)<<f)|((1<<f)-1);b[0][0]=a[0][0]
            a[1][0]=1;b[0][1]=1
            a[2][0]=(((1<<e)-1)<<f)|1;b[0][2]=to_bits(Fraction(1),fmt)
        scales=None;source_b=b;dequant_flags=0;quant_group_log2=group_log2 if original else 2
        if floating_mixed:
            source_b=[[rng.randrange(16)for _ in range(n)]for _ in range(k)]
            source_b[0][0]=8
            groups=(k+(1<<quant_group_log2)-1)>>quant_group_log2
            coefficients=[Fraction(0),Fraction(1,3),Fraction(1,1<<25),Fraction(1<<16),Fraction(10**38)]
            scales=[[to_bits(coefficients[(g+j)%len(coefficients)],Format.F32)for j in range(n)]for g in range(groups)]
            if original:
                _,source_b,_,scales=quantize(*original,Format.I4,1<<quant_group_log2)
            b=[]
            for t,row in enumerate(source_b):
                values=[]
                for j,code in enumerate(row):
                    product,ff=fp_operation(to_bits(signed(code,4),Format.F32),scales[t>>quant_group_log2][j],0,Format.F32,Op.MUL)
                    value,nf=narrow32(product,fmt);values.append(value);dequant_flags|=ff|nf
                b.append(values)
        initial=None
        if accumulate:
            initial=[[( ((-1 if (i+j)%2 else 1)*(1<<(aw-3))+rng.randrange(-9,10))&((1<<aw)-1)
                if fmt<Format.F4 else to_bits(Fraction(i-j,4),Format.F64 if fmt==Format.F64 else Format.F32))
                for j in range(n)]for i in range(m)]
        began=time.perf_counter()
        if reference_backend=='cpu':
            from simd_eval.cpu_reference import gemm as cpu_gemm
            reference,flags,reference_evidence=cpu_gemm(a,b,fmt,initial)
        else:reference,flags=gemm(a,b,fmt,initial=initial,return_flags=True)
        reference,output_flags,output_clips=finish_matrix(reference,fmt,output)
        output_width=WIDTHS[Format[output_format]]if output else aw
        reference_seconds+=time.perf_counter()-began
        addr_a=0x10000000+index*0x100000;addr_b=0x30000000+index*0x100000
        addr_c=0x80000000+index*0x100000+(index%64)
        astride=((k*width+63)//64)*8+8;bstride=((n*weight_width+63)//64)*8+16
        cstride=(n*output_width+7)//8+13
        if index==chain_begin:addr_c-=index%64;cstride=((n*ab+7)//8)*8
        if dependent:addr_a=previous['address'];astride=previous['stride']
        def tensor(base,stride,matrix,bits):
            data=bytearray(len(matrix)*stride)
            for row,values in enumerate(matrix):
                raw=sum((v&((1<<bits)-1))<<(j*bits)for j,v in enumerate(values))
                size=(len(values)*bits+7)//8
                data[row*stride:row*stride+size]=raw.to_bytes(size,'little')
            for pos in range(0,len(data),16):
                commands.append(f'L {base+pos} {int.from_bytes(data[pos:pos+16],"little"):032x}\n')
        # For the dependent job, A is the previous RTL writeback in external
        # memory. The oracle's reference is never written into that memory.
        if not dependent:tensor(addr_a,astride,a,width)
        if not floating_mixed:tensor(addr_b,bstride,b,weight_width)
        if floating_mixed:
            # Only compressed codes and scale bits enter simulated memory.
            # The oracle's dequantized weights are never supplied to RTL.
            tensor(addr_b,bstride,source_b,4)
        scale_address=0x60000000+index*0x100000
        scale_stride=((n*4+7)//8)*8+8
        if scales:tensor(scale_address,scale_stride,scales,32)
        if accumulate:tensor(addr_c,cstride,initial,aw)
        if index==chain_begin:previous=dict(reference=reference,address=addr_c,stride=cstride)
        jobs[index]=dict(fmt=fmt.name,operand_mode='W4A16'if floating_mixed else'W4A8'if weight4 else'symmetric',
                        accumulate=accumulate,m=m,n=n,k=k,flags=dequant_flags|output_flags,
                        expected_output_saturations=output_clips,
                        logical_input_bytes=m*((k*width+7)//8)+k*((n*weight_width+7)//8),
                        logical_scale_bytes=(len(scales)*n*4)if scales else 0,
                        initial_c_read_bytes=m*n*ab if accumulate else 0,logical_output_bytes=m*((n*output_width+7)//8),
                        expected_dequant_operations=k*n*((m+rows-1)//rows)if floating_mixed else 0)
        for i in range(m):
            for j in range(n):
                jobs[index]['flags']|=flags[i][j]
            for v,byte in enumerate(packed_row(reference[i],output_width)):
                expected[addr_c+i*cstride+v]=byte
        commands.append(f'G {int(fmt)} {m} {n} {k} {addr_a} {addr_b} {addr_c} {astride} {bstride} {cstride} {index} {int(weight4)} {int(accumulate)} {scale_address} {scale_stride} {quant_group_log2}\n')
        total_macs+=m*n*k
    trace=out/'commands.txt';trace.write_text(''.join(commands))
    with (out/'simulation.log').open('w')as log:
        proc=subprocess.Popen([str(dest/'sim'),str(trace),str(core_ps),str(memory_ps),
            str(config_path)if dram else'-',str(out/'dram'),'207'],stdout=log,stderr=subprocess.STDOUT,text=True)
        try:status=proc.wait(timeout=timeout if timeout is not None else max(1800,120*rows*cols))
        except subprocess.TimeoutExpired:
            proc.kill();proc.wait();raise
    stdout=(out/'simulation.log').read_text()
    if status:raise RuntimeError(stdout[-3000:])
    written={};done={};summary=None;macs=0;compute=0;onchip=None;accepted_start=None
    for line in stdout.splitlines():
        if line.startswith('W '):
            _,address,value=line.split();written[int(address)]=int(value)
        if line.startswith('D '):
            fields=list(map(int,line.split()[1:]))
            if len(fields)not in(8,11,13)or(caps.get('utilization_counters')and len(fields)<11):
                raise AssertionError('missing descriptor utilization counters')
            tag,flags,cycles,count,overflow,memory_cycles,reads,ps=fields[:8];done[tag]=True
            if flags!=jobs[tag]['flags']:raise AssertionError(f'flags {tag}: {flags} != {jobs[tag]["flags"]}')
            if count!=jobs[tag]['m']*jobs[tag]['n']*jobs[tag]['k']:raise AssertionError('hardware MAC count')
            if overflow:raise AssertionError('unexpected overflow')
            jobs[tag].update(compute_cycles=cycles,memory_cycles=memory_cycles,read_bursts=reads)
            if len(fields)>=11:
                jobs[tag].update(utilization(*fields[8:11],cycles,count,rows*cols,64//WIDTHS[Format[jobs[tag]['fmt']]]))
            if caps.get('w4a16_dequant'):
                if len(fields)!=13 or fields[11]!=jobs[tag]['expected_dequant_operations']:
                    raise AssertionError('incorrect hardware dequantization operation count')
                jobs[tag].update(dequant_operations=fields[11],dequant_memory_cycles=fields[12])
            compute+=cycles;macs+=count
        if line.startswith('S '):summary=list(map(int,line.split()[1:]))
        if line.startswith('H '):onchip=list(map(int,line.split()[1:]))
        if line.startswith('T '):
            if accepted_start is not None:raise AssertionError('duplicate first-command timestamp')
            _,first_tag,accepted_start=line.split();accepted_start=int(accepted_start)
    require_output_bytes(written,expected)
    if len(done)!=len(jobs)or macs!=total_macs or summary is None:raise AssertionError('missing command completions')
    start,submitted,end,read_requests,write_requests,write_bytes,read_bursts,write_bursts=summary
    traffic=dram_evidence(stdout,read_requests,write_requests,read_bursts,write_bursts,dram)
    if accepted_start is not None and start!=accepted_start:raise AssertionError('end-to-end timer did not start at descriptor handshake')
    if sum(j['read_bursts']for j in jobs.values())!=read_requests or write_bytes!=len(expected):raise AssertionError('memory accounting')
    metrics=dict(passed=True,execution_path='autonomous_rtl_row_major_gemm',input_schedule='descriptor_sequential_gather',stages=stages,rows=rows,cols=cols,
        test_scope='workload'if original else'smoke'if smoke else'full',sram_read_latency=caps.get('sram_read_latency',1),
        dependent_descriptors=not bool(original),
        initial_c_dma=bool(caps.get('gemm_initial_c_dma')),
        native_w4a8=True,
        w4a16_dequant=bool(caps.get('w4a16_dequant')),
        descriptors=len(jobs),formats=sorted({j['fmt']for j in jobs.values()}),verified_output_bytes=len(written),exact_output_match=True,
        macs=macs,compute_cycles=compute,elapsed_ps=end-start,submitted_ps=submitted-start,
        elapsed_time_origin='first_descriptor_acceptance'if accepted_start is not None else'simulation_after_reset',
        core_frequency_hz=1e12/core_ps,memory_frequency_hz=1e12/memory_ps,
        dma_read_requests=read_requests,dma_write_requests=write_requests,output_payload_bytes=write_bytes,
        dram_read_bursts=read_bursts,dram_write_bursts=write_bursts,dram_physical_bytes=traffic['dram_physical_bytes'],
        dram=dram,timing_validated=False,time_kind='configured_clock_projection',
        rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'],testbench_sha256=caps['testbench_sha256'],
        dram_library_sha256=caps['dram_library_sha256'],trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
    metrics['dram_config_sha256']=hashlib.sha256(config_path.read_bytes()).hexdigest()if dram else None
    metrics['dma_backend']=caps.get('dma_backend','rtl')
    metrics['output_sha256']=hashlib.sha256(bytes(written[a]for a in sorted(written))).hexdigest()
    metrics['hardware_contract']=caps.get('hardware_contract',hardware_contract(caps))
    for field in('logical_input_bytes','logical_scale_bytes','initial_c_read_bytes','logical_output_bytes'):
        metrics[field]=sum(job[field]for job in jobs.values())
    metrics['useful_transfer_bytes']=sum(metrics[field]for field in
        ('logical_input_bytes','logical_scale_bytes','initial_c_read_bytes','logical_output_bytes'))
    metrics['useful_bandwidth_GB_s']=metrics['useful_transfer_bytes']/((end-start)*1e-12)/1e9
    metrics['gmac_s']=macs/((end-start)*1e-12)/1e9
    metrics.update(traffic)
    metrics['dma_transfer_bytes']=64*(read_requests+write_requests)
    metrics['dma_bandwidth_GB_s']=metrics['dma_transfer_bytes']/((end-start)*1e-12)/1e9
    if caps.get('onchip_counters') and onchip is None:raise AssertionError('missing on-chip counters')
    if onchip:
        metrics.update(sram_read_bytes=onchip[0],sram_write_bytes=onchip[1],onchip_moved_bytes=sum(onchip))
        if len(onchip)==4:metrics.update(aggregate_read_bytes=onchip[2],aggregate_write_bytes=onchip[3])
    if caps.get('utilization_counters'):
        capacity=sum(j['compute_cycles']*rows*cols*(64//WIDTHS[Format[j['fmt']]])for j in jobs.values())
        vector_lanes=sum(j['issued_pe_vectors']*(64//WIDTHS[Format[j['fmt']]])for j in jobs.values())
        issued=sum(j['issued_pe_vectors']for j in jobs.values())
        metrics.update(issued_pe_vectors=issued,idle_pe_cycles=compute*rows*cols-issued,
            array_issue_utilization=issued/(compute*rows*cols),lane_utilization=macs/vector_lanes,
            mac_utilization=macs/capacity,
            feedback_stall_cycles=sum(j['feedback_stall_cycles']for j in jobs.values()),
            input_wait_cycles=sum(j['input_wait_cycles']for j in jobs.values()))
    if caps.get('w4a16_dequant'):
        metrics.update(dequant_operations=sum(j['dequant_operations']for j in jobs.values()),
                       dequant_memory_cycles=sum(j['dequant_memory_cycles']for j in jobs.values()))
    if original:
        exact=[[sum(original[0][i][t]*original[1][t][j]for t in range(k))for j in range(n)]for i in range(m)]
        actual=read_matrix(written,addr_c,cstride,m,n,output_width)
        metrics.update(format=fmt.name,operand_mode=mode,m=m,n=n,k=k,workload_sha256=workload_hash(*original),
                       group_size=(1<<group_log2)if mode=='W4A16'else None,
                       gops=2*macs/((end-start)*1e-12)/1e9,
                       dram_bandwidth_GB_s=traffic['dram_physical_bytes']/((end-start)*1e-12)/1e9 if dram else None,
                       source_npz_sha256=hashlib.sha256(Path(npz).read_bytes()).hexdigest())
        error_format=Format[output_format]if output else Format.F32
        error_options=dict(scale=Fraction(output['output_scale_exact']),zero_point=output_zero_point)if output else{}
        metrics.update(numerical_errors(actual,exact,error_format,**error_options))
        if framework_reference is not None:
            metrics.update({'framework_'+key:value for key,value in numerical_errors(actual,framework_reference,error_format,**error_options).items()})
    if output:metrics.update(output)
    metrics.update(epilogue_evidence(stdout,caps,jobs,bool(output)))
    metrics.update(sequencer_cycle_evidence(stdout,caps,jobs))
    metrics.update(native_prefetch_evidence(stdout,caps,jobs))
    metrics['native_prefetch_requested']=bool(prefetch and caps.get('autonomous_prefetch'))
    if metrics['native_prefetch_requested']:metrics['input_schedule']='descriptor_wave_fifo_sram_prefetch'
    metrics.update(dma_link_metrics(metrics,read_requests,write_requests))
    metrics.update(reference_evidence);metrics['reference_host_seconds']=reference_seconds
    metrics.update({key:caps[key] for key in ('process','architecture_sha256','dependency_lock_sha256','sram_macro','storage_implementation','sram_physical_certified')})
    write_metrics(out,metrics);(out/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n');print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stages',type=int,default=2);p.add_argument('--rows',type=int,default=2)
    p.add_argument('--memory-ps',type=int,default=MEMORY_PS)
    p.add_argument('--reference-backend',choices=['rational','cpu'],default='rational')
    p.add_argument('--no-prefetch',dest='prefetch',action='store_false',help='serialize next-tile loading on the same native hardware')
    p.add_argument('--cols',type=int,help='defaults to --rows');p.add_argument('--dram',action='store_true');p.add_argument('--out');p.add_argument('--core-ps',type=int,default=CORE_PS);p.add_argument('--smoke',action='store_true')
    p.add_argument('--npz');p.add_argument('--format',choices=['F16','BF16'],default='BF16');p.add_argument('--mode',choices=['symmetric','W4A16'],default='symmetric');p.add_argument('--group-log2',type=int,default=5);p.add_argument('--build-dir')
    p.add_argument('--output-format',choices=[fmt.name for fmt in Format]);p.add_argument('--output-scale',default='1',help='multiply by this FP32 scale before output RNE conversion')
    p.add_argument('--output-zero-point',type=int,default=0)
    p.add_argument('--timeout',type=int,help='host runtime limit in seconds; default scales with PE count');p.add_argument('--dram-config');a=p.parse_args()
    check(a.stages,a.rows,a.cols if a.cols is not None else a.rows,a.dram,a.out,a.core_ps,a.smoke,a.npz,a.format,a.mode,a.group_log2,a.build_dir,a.timeout,a.dram_config,a.memory_ps,a.reference_backend,a.prefetch,a.output_format,a.output_scale,a.output_zero_point)
