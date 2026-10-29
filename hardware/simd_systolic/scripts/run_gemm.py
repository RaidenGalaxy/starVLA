#!/usr/bin/env python3
"""Drive arbitrary tiled GEMM through RTL SRAM/CDC and optional live DRAMSim3."""
import argparse,csv,hashlib,json,random,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
import sys as _architecture_sys
_architecture_sys.path.insert(0,str(ROOT/'python'))
from simd_eval.architecture import CORE_PS, MEMORY_PS
from simd_eval.reporting import workload_hash,write_metrics,verify_timing,memory_evidence,numerical_errors,utilization_evidence,dram_geometry,dma_link_metrics
from runtime import validated_build
from simd_eval.reference import Format,WIDTHS,ACC_WIDTHS,gemm,pack,to_bits,signed,decode

def run(fmt='I4',m=7,n=19,k=65,stages=2,rows=2,cols=2,dram=False,
        core_ps=CORE_PS,memory_ps=MEMORY_PS,out=None,npz=None,timing=None,serialize_loads=False,build_dir=None):
    fmt=Format[fmt];w=WIDTHS[fmt];aw=ACC_WIDTHS[fmt];lanes=64//w
    if not 1<=k<=65536:raise ValueError('K must be in [1,65536]')
    dest=Path(build_dir)if build_dir else ROOT/f'.build/system_{stages}_{rows}x{cols}'
    build_guard,caps=validated_build(dest)
    if(caps['target'],caps['stages'],caps['rows'],caps['cols'])!=('system',stages,rows,cols):raise ValueError('system build scope differs from requested GEMM')
    out=Path(out)if out else ROOT/f'artifacts/gemm_{fmt.name}_{stages}_{m}x{n}x{k}_{"dram"if dram else"ideal"}'
    out.mkdir(parents=True,exist_ok=True)
    workload_id=None
    original=None
    if npz:
        import numpy as np
        f=np.load(npz,allow_pickle=False);aa=f['a'];bb=f['b']
        if aa.ndim!=2 or bb.ndim!=2 or aa.shape[1]!=bb.shape[0]:raise ValueError('invalid A[M,K]/B[K,N] shapes')
        m,k=aa.shape;n=bb.shape[1]
        original=(aa.tolist(),bb.tolist())
        workload_id=workload_hash(*original)
        if k>65536:raise ValueError('K exceeds hardware bound')
        if 'raw_bits' in f and bool(f['raw_bits']):
            a=aa.tolist();b=bb.tolist();original=None;workload_id=None
        else:
            def convert(x):
                if fmt<Format.F4:
                    value=int(x)
                    if value!=x or not -(1<<(w-1))<=value<(1<<(w-1)):
                        raise ValueError('integer inputs must already be quantized into the signed source range')
                    return value&((1<<w)-1)
                return to_bits(Fraction(float(x)),fmt)
            a=[[convert(x)for x in row]for row in aa];b=[[convert(x)for x in row]for row in bb]
    else:
        rng=random.Random(207)
        def number():
            x=rng.randrange(-2,3)
            return x&((1<<w)-1)if fmt<Format.F4 else to_bits(Fraction(x,2),fmt)
        a=[[number()for _ in range(k)]for _ in range(m)]
        b=[[number()for _ in range(n)]for _ in range(k)]
    if min(m,n,k)<1:raise ValueError('M, N and K must be positive')
    expected,expected_flags=gemm(a,b,fmt,return_flags=True);commands=[];reads={};read_flags={};tag=0;total_payload=0;expected_macs=m*n*k
    tiles=[(i,j)for i in range(0,m,rows)for j in range(0,n,cols*lanes)]
    # Contexts in one command share the same row mask; group by M tile.
    for i in range(0,m,rows):
      ntiles=[j for ii,j in tiles if ii==i]
      for group_begin in range(0,len(ntiles),16):
        group=ntiles[group_begin:group_begin+16];contexts=len(group)
        # Tail masks differ between contexts, while the tile command has one
        # mask vector. Handle the last partial N tile as a separate command group.
        if len(group)>1 and group[-1]+cols*lanes>n:
            # Recurse into grouping without changing descriptor semantics below.
            groups=[group[:-1],group[-1:]]
        else:groups=[group]
        for group in groups:
          if not group:continue
          contexts=len(group);row_mask=(1<<min(rows,m-i))-1
          masks=[(1<<max(0,min(lanes,n-group[-1]-c*lanes)))-1 for c in range(cols)]
          for begin in range(0,k,8):
            end=min(k,begin+8);av=[];bv=[]
            for t in range(begin,end):
              for j in group:
                aa=[a[i+r][t]if i+r<m else 0 for r in range(rows)]
                bb=[pack([b[t][j+c*lanes+l]if j+c*lanes+l<n else 0 for l in range(lanes)],w)for c in range(cols)]
                # Each row/column slot is a 64-bit SRAM word; report all padding.
                av.extend(aa+[0]*(len(aa)%2));bv.extend(bb+[0]*(len(bb)%2))
            for bank,words in [(0,av),(1,bv)]:
              for p in range(0,len(words),2):
                commands.append(f'L {bank} {p//2} {pack(words[p:p+2],64):032x}\n')
            commands.append(f'C {int(fmt)} {contexts} {end-begin} {int(begin==0)} {row_mask:x} '+
                ' '.join(f'{x:x}'for x in masks)+f' {tag}\n');tag+=1
            if serialize_loads:commands.append('D\n')
          commands.append('D\n')
          for ctx,j in enumerate(group):
            for r in range(rows):
              if i+r>=m:continue
              for c in range(cols):
                if j+c*lanes>=n:continue
                value=pack([expected[i+r][j+c*lanes+l]if i+r<m and j+c*lanes+l<n else 0 for l in range(lanes)],aw)
                flags=0
                for l in range(lanes):
                    if j+c*lanes+l<n:flags|=expected_flags[i+r][j+c*lanes+l]
                read_flags[tag]=flags
                reads[tag]=value;commands.append(f'R {ctx} {r} {c} {tag}\n');tag+=1
                count=max(0,min(lanes,n-j-c*lanes))if i+r<m else 0
                if count:
                    size=(count*aw+7)//8;commands.append(f'O {size}\n');total_payload+=size
    trace=out/'commands.txt';trace.write_text(''.join(commands))
    config=str(ROOT/'configs/ddr4.ini')if dram else '-'
    command=[str(dest/'sim'),str(trace),str(core_ps),str(memory_ps),config,str(out/'dramsim3'),'207']
    started=time.monotonic();result=subprocess.run(command,text=True,capture_output=True,timeout=1800)
    (out/'simulation.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(result.stdout[-3000:]+result.stderr)
    seen=set();summary=None;outputs={}
    for line in result.stdout.splitlines():
        if line.startswith('R '):
            _,t,v,flags,ps=line.split();t=int(t);seen.add(t);outputs[t]=int(v,16)
            if int(flags)!=read_flags[t]:raise AssertionError(f'exception flags {t}: expected {read_flags[t]}, actual {flags}')
            if outputs[t]!=reads[t]:raise AssertionError(f'read {t}: expected {reads[t]:x}, actual {v}')
        if line.startswith('S '):summary=list(map(int,line.split()[1:]))
    if len(seen)!=len(reads)or summary is None:raise AssertionError('missing results/summary')
    start,submitted,end,compute_cycles,macs,load_bytes,write_bytes,read_bursts,write_bursts,aux_count,aux_ps,overflows,exceptions=summary
    if macs!=expected_macs:raise AssertionError((macs,expected_macs))
    proof=json.loads(Path(timing).read_text())if timing else None
    validated=verify_timing(proof,caps,stages,rows,cols,core_ps)
    seconds=(end-start)*1e-12;physical=(read_bursts+write_bursts)*dram_geometry(result.stdout)['dram_burst_bytes']
    metrics={'schema_version':1,'format':fmt.name,'m':m,'n':n,'k':k,'stages':stages,'rows':rows,'cols':cols,
      'input_schedule':'tile_barrier'if serialize_loads else'sram_wave_fifo_overlap',
      'workload_sha256':workload_id,'rtl_sha256':caps['rtl_sha256'],'timing_validated':validated,'time_kind':'timing_validated'if validated else'configured_clock_projection',
      'core_frequency_hz':1e12/core_ps,'memory_frequency_hz':1e12/memory_ps,
      'elapsed_ps':end-start,'submitted_ps':submitted-start,'compute_cycles':compute_cycles,
      'projected_seconds':seconds,'compute_seconds':compute_cycles*core_ps*1e-12,
      'macs':macs,'operations':2*macs,'gmac_per_s':macs/seconds/1e9,'gops':2*macs/seconds/1e9,
      'auxiliary_operations':aux_count,'auxiliary_elapsed_ps':aux_ps,
      'overflow_lane_events':overflows,'exception_vector_operations':exceptions,
      'dma_payload_bytes':load_bytes+write_bytes,'logical_input_bytes':m*((k*w+7)//8)+k*((n*w+7)//8),
      'output_payload_bytes':write_bytes,'dram_read_bursts':read_bursts,'dram_write_bursts':write_bursts,
      'dram_physical_bytes':physical,'dram_bandwidth_GB_s':physical/seconds/1e9 if dram else None,
      'dma_bandwidth_GB_s':(load_bytes+write_bytes)/seconds/1e9,'dram':dram,
      'exact_output_match':True,'verified_reads':len(reads),'host_wall_seconds':time.monotonic()-started,
      'random_seed':207,'dma_width_bits':128,'max_outstanding':32,
      'dram_config_sha256':hashlib.sha256((ROOT/'configs/ddr4.ini').read_bytes()).hexdigest()if dram else None,
      'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest()}
    metrics.update(memory_evidence(result.stdout,caps,read_bursts,write_bursts,load_bytes,write_bytes,dram))
    metrics['logical_output_bytes']=m*n*(aw//8)
    metrics['useful_transfer_bytes']=metrics['logical_input_bytes']+metrics['logical_output_bytes']
    metrics['useful_bandwidth_GB_s']=metrics['useful_transfer_bytes']/seconds/1e9
    metrics.update(utilization_evidence(result.stdout,caps,compute_cycles,macs,rows*cols,lanes))
    if original and fmt>=Format.F4:
        source_a,source_b=original
        reference=[[sum(Fraction(float(source_a[i][t]))*Fraction(float(source_b[t][j]))for t in range(k))
                    for j in range(n)]for i in range(m)]
        metrics.update(numerical_errors(expected,reference,Format.F64 if fmt==Format.F64 else Format.F32))
    metrics.update(dma_link_metrics(metrics,metrics['rtl_read_bursts'],metrics['rtl_write_bursts']))
    metrics.update({key:caps[key] for key in ('process','architecture_sha256','dependency_lock_sha256','sram_macro','storage_implementation','sram_physical_certified')})
    write_metrics(out,metrics)
    print(json.dumps(metrics,indent=2));return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--format',choices=[f.name for f in Format],default='I4')
    for name,value in [('m',7),('n',19),('k',65),('stages',2),('rows',2),('cols',2),('core-ps',CORE_PS),('memory-ps',MEMORY_PS)]:
        p.add_argument('--'+name,type=int,default=value)
    p.add_argument('--dram',action='store_true');p.add_argument('--out');p.add_argument('--npz');p.add_argument('--timing')
    p.add_argument('--serialize-loads',action='store_true',help='ablate DMA/compute overlap by waiting after each tile')
    p.add_argument('--build-dir')
    a=p.parse_args();run(a.format,a.m,a.n,a.k,a.stages,a.rows,a.cols,a.dram,a.core_ps,a.memory_ps,a.out,a.npz,a.timing,a.serialize_loads,a.build_dir)
