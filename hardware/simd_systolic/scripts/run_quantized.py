#!/usr/bin/env python3
"""Execute group partials, I2F, SRAM scales and FP32 aggregation in RTL.

@tag references in the command stream use actual retired hardware results;
the rational reference is only used to check them, never to feed accumulators.
"""
import argparse,hashlib,json,math,random,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
import sys as _architecture_sys
_architecture_sys.path.insert(0,str(ROOT/'python'))
from simd_eval.architecture import CORE_PS, MEMORY_PS
from simd_eval.reporting import workload_hash,write_metrics,memory_evidence,numerical_errors,utilization_evidence,dram_geometry,dma_link_metrics
from runtime import validated_build
from simd_eval.reference import Format,Op,WIDTHS,ACC_WIDTHS,gemm,pack,operate,to_bits,decode
from simd_eval.quantization import quantize,load_codes

def run(fmt='I4',m=3,n=5,k=17,group_size=5,stages=2,rows=2,cols=2,dram=True,npz=None,out=None,core_ps=None,memory_ps=MEMORY_PS,quantized_npz=None):
    core_ps=core_ps or CORE_PS
    fmt=Format[fmt];w=WIDTHS[fmt];aw=ACC_WIDTHS[fmt];lanes=64//w
    if group_size<1:raise ValueError('group size must be positive')
    if npz and quantized_npz:raise ValueError('select original tensors or encoded checkpoint input')
    if quantized_npz:
        a,b,qa,qb,sa,sb,group_size=load_codes(quantized_npz,fmt)
        m=len(a);k=len(a[0]);n=len(b[0])
    elif npz:
        import numpy as np
        data=np.load(npz,allow_pickle=False)
        a=[[Fraction(float(x))for x in row]for row in data['a']]
        b=[[Fraction(float(x))for x in row]for row in data['b']]
        m=len(a);k=len(a[0]);n=len(b[0])
    else:
        rng=random.Random(301)
        def number():return Fraction(round(rng.gauss(0,1),6))
        a=[[number()for _ in range(k)]for _ in range(m)];b=[[number()for _ in range(n)]for _ in range(k)]
    if not 1<=k<=65536:raise ValueError('K outside supported bound')
    workload_id=workload_hash(a,b) if npz or quantized_npz else None
    if not quantized_npz:qa,qb,sa,sb=quantize(a,b,fmt,group_size)
    groups=(k+group_size-1)//group_size
    partials=[gemm([row[g*group_size:min(k,(g+1)*group_size)]for row in qa],
        qb[g*group_size:min(k,(g+1)*group_size)],fmt,return_flags=True)for g in range(groups)]
    out=Path(out)if out else ROOT/f'artifacts/quant_{fmt.name}_{stages}_{m}x{n}x{k}_g{group_size}'
    out.mkdir(parents=True,exist_ok=True)
    commands=[];expected={};tag=0;fp_sum=[[0]*n for _ in range(m)]
    def result(value,flags=0):
        nonlocal tag
        t=tag;tag+=1;expected[t]=(value,flags);return t
    for i in range(0,m,rows):
      tiles=list(range(0,n,cols*lanes))
      batches=[tiles[t:t+16]for t in range(0,len(tiles),16)]
      batches=[part for batch in batches for part in ([batch[:-1],batch[-1:]]if len(batch)>1 and batch[-1]+cols*lanes>n else[batch])if part]
      for batch in batches:
        contexts=len(batch);row_mask=(1<<min(rows,m-i))-1
        masks=[(1<<max(0,min(lanes,n-batch[-1]-c*lanes)))-1 for c in range(cols)]
        for g in range(groups):
          lo=g*group_size;hi=min(k,lo+group_size)
          for begin in range(lo,hi,8):
            end=min(hi,begin+8);av=[];bv=[]
            for t in range(begin,end):
              for j in batch:
                aa=[qa[i+r][t]if i+r<m else 0 for r in range(rows)]
                bb=[pack([qb[t][j+c*lanes+l]if j+c*lanes+l<n else 0 for l in range(lanes)],w)for c in range(cols)]
                av+=aa+[0]*(len(aa)%2);bv+=bb+[0]*(len(bb)%2)
            for bank,words in [(0,av),(1,bv)]:
              for p in range(0,len(words),2):commands.append(f'L {bank} {p//2} {pack(words[p:p+2],64):032x}\n')
            commands.append(f'C {int(fmt)} {contexts} {end-begin} {int(begin==lo)} {row_mask:x} '+
                ' '.join(f'{v:x}'for v in masks)+f' {tag}\n');tag+=1
          commands.append('D\n')
          for ctx,j in enumerate(batch):
            for r in range(min(rows,m-i)):
              for c in range(cols):
                count=max(0,min(lanes,n-j-c*lanes))
                if not count:continue
                part=[partials[g][0][i+r][j+c*lanes+l]if l<count else 0 for l in range(lanes)]
                raw_flags=0
                for v in range(count):raw_flags|=partials[g][1][i+r][j+c*lanes+v]
                raw=pack(part,aw);raw_tag=result(raw,raw_flags)
                commands.append(f'R {ctx} {r} {c} {raw_tag}\n')
                if fmt<Format.F4:
                    converted,flags,_=operate(0,0,raw,fmt,Op.I2F,(1<<count)-1)
                    converted_tag=result(converted,flags)
                    commands.append(f'A {int(fmt)} 7 0 0 @{raw_tag} {(1<<count)-1:x} {converted_tag} 0 0 0 0 0\n')
                else:converted=raw;converted_tag=raw_tag
                for l in range(0,count,2):
                    valid=min(2,count-l);mask=(1<<valid)-1
                    scale_a=[sa[i+r][g]]*valid
                    scale_b=[sb[g][j+c*lanes+l+v]for v in range(valid)]
                    scale_word=pack([item for pair in zip(scale_a,scale_b)for item in pair],32)
                    commands.append(f'L 2 0 {scale_word:032x}\n')
                    gamma,gflags,_=operate(pack(scale_a,32),pack(scale_b,32),0,Format.F32,Op.MUL,mask)
                    gamma_tag=result(gamma,gflags)
                    commands.append(f'A 9 0 0 0 0 {mask:x} {gamma_tag} 0 0 0 1 0\n')
                    old=pack([fp_sum[i+r][j+c*lanes+l+v]for v in range(valid)],32)
                    converted_pair=converted>>(l*32)&((1<<64)-1)
                    summed,sflags,_=operate(converted_pair,gamma,old,Format.F32,Op.MAC,mask)
                    summed_tag=result(summed,sflags)
                    index=(((ctx*rows+r)*cols+c)*8+l//2)
                    commands.append(f'A 9 1 @{converted_tag}:{l*32}:64 @{gamma_tag}:0:64 0 {mask:x} {summed_tag} 1 {index} {int(g==0)} 0 0\n')
                    for v in range(valid):fp_sum[i+r][j+c*lanes+l+v]=summed>>(32*v)&0xffffffff
                    if g==groups-1:commands.append(f'O {4*valid}\n')
    trace=out/'commands.txt';trace.write_text(''.join(commands));dest=ROOT/f'.build/system_{stages}_{rows}x{cols}'
    build_guard,caps=validated_build(dest)
    started=time.monotonic()
    proc=subprocess.run([str(dest/'sim'),str(trace),str(core_ps),str(memory_ps),
      str(ROOT/'configs/ddr4.ini')if dram else'-',str(out/'dram'),'207'],capture_output=True,text=True,timeout=1800)
    (out/'simulation.log').write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError(proc.stdout[-2000:]+proc.stderr)
    seen=set();summary=None
    for line in proc.stdout.splitlines():
        if line.startswith('R '):
            _,t,v,f,ps=line.split();t=int(t);seen.add(t)
            if (int(v,16),int(f))!=expected[t]:raise AssertionError(f'tag {t}: {(v,f)} != {tuple(hex(x)for x in expected[t])}')
        if line.startswith('S '):summary=list(map(int,line.split()[1:]))
    assert len(seen)==len(expected)
    start,submitted,end,cycles,macs,load_bytes,write_bytes,rb,wb,aux_count,aux_ps,overflows,exceptions=summary
    assert macs==m*n*k
    original=[[sum(a[ii][t]*b[t][jj]for t in range(k))for jj in range(n)]for ii in range(m)]
    metrics={'format':fmt.name,'execution_path':'symmetric_group_quantization','input_schedule':'sram_wave_fifo_overlap','m':m,'n':n,'k':k,
      'group_size':group_size,'groups':groups,'stages':stages,'rows':rows,'cols':cols,
      'elapsed_ps':end-start,'submitted_ps':submitted-start,'array_compute_cycles':cycles,
      'macs':macs,'auxiliary_operations':aux_count,'auxiliary_elapsed_ps':aux_ps,
      'overflow_lane_events':overflows,'exception_vector_operations':exceptions,
      'dma_payload_bytes':load_bytes+write_bytes,'dram_read_bursts':rb,'dram_write_bursts':wb,
      'dram_physical_bytes':dram_geometry(proc.stdout)['dram_burst_bytes']*(rb+wb),'scale_storage_dtype':'FP32',
      'exact_hardware_reference_match':True,'verified_reads':len(seen),'dram':dram,
      'core_frequency_hz':1e12/core_ps,'memory_frequency_hz':1e12/memory_ps,'timing_validated':False,
      'workload_sha256':workload_id,'gops':2*macs/((end-start)*1e-12)/1e9,
      'compute_cycles':cycles,'dma_bandwidth_GB_s':(load_bytes+write_bytes)/((end-start)*1e-12)/1e9,
      'dram_bandwidth_GB_s':dram_geometry(proc.stdout)['dram_burst_bytes']*(rb+wb)/((end-start)*1e-12)/1e9 if dram else None,
      'time_kind':'configured_clock_projection','host_wall_seconds':time.monotonic()-started,
      'rtl_sha256':caps['rtl_sha256'],'random_seed':207,'dma_width_bits':128,'max_outstanding':32,
      'dram_config_sha256':hashlib.sha256((ROOT/'configs/ddr4.ini').read_bytes()).hexdigest()if dram else None,
      'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest()}
    metrics.update(memory_evidence(proc.stdout,caps,rb,wb,load_bytes,write_bytes,dram))
    metrics.update(utilization_evidence(proc.stdout,caps,cycles,macs,rows*cols,64//WIDTHS[fmt]))
    metrics.update(numerical_errors(fp_sum,original,Format.F32))
    metrics['quantized_source_sha256']=hashlib.sha256(Path(quantized_npz).read_bytes()).hexdigest()if quantized_npz else None
    metrics.update(dma_link_metrics(metrics,metrics['rtl_read_bursts'],metrics['rtl_write_bursts']))
    metrics.update({key:caps[key] for key in ('process','architecture_sha256','dependency_lock_sha256','sram_macro','storage_implementation','sram_physical_certified')})
    write_metrics(out,metrics)
    (out/'output_fp32_bits.json').write_text(json.dumps(fp_sum)+'\n');print(json.dumps(metrics,indent=2));return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--format',choices=['I4','I8','F4','F8'],default='I4')
    for name,value in [('m',3),('n',5),('k',17),('group-size',5),('stages',2),('rows',2),('cols',2)]:p.add_argument('--'+name,type=int,default=value)
    p.add_argument('--ideal',action='store_true');p.add_argument('--npz');p.add_argument('--quantized-npz');p.add_argument('--out');p.add_argument('--core-ps',type=int);p.add_argument('--memory-ps',type=int,default=MEMORY_PS);a=p.parse_args()
    run(a.format,a.m,a.n,a.k,a.group_size,a.stages,a.rows,a.cols,not a.ideal,a.npz,a.out,a.core_ps,a.memory_ps,a.quantized_npz)
