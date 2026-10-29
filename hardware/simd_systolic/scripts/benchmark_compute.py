#!/usr/bin/env python3
"""Same-shape data-ready GEMM; exclude observed readout cycles from compute."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
import sys as _architecture_sys
_architecture_sys.path.insert(0,str(ROOT/'python'))
from simd_eval.architecture import CORE_PS, MEMORY_PS
from runtime import validated_build
from simd_eval.reference import Format,WIDTHS,ACC_WIDTHS,to_bits,pack
from simd_eval.reporting import workload_hash,write_metrics,verify_timing
from simd_eval.area import load_area,area_metrics

def run(stages,rows,cols,n,k,core_ps,out,build_dir=None,area=None,timing=None):
    dest=Path(build_dir)if build_dir else ROOT/f'.build/array_{stages}_{rows}x{cols}';guard,caps=validated_build(dest)
    if (caps['stages'],caps['rows'],caps['cols'])!=(stages,rows,cols):raise ValueError('build scope differs from benchmark')
    proof=json.loads(Path(timing).read_text())if timing else None
    timing_validated=verify_timing(proof,caps,stages,rows,cols,core_ps)
    area_evidence=None
    if area:
        area_evidence=load_area(area)
    if n%(cols*16) or n<cols*16:raise ValueError('N must be a positive multiple of cols*16')
    if not 1<=k<=65536 or core_ps<=0:raise ValueError('invalid K or core clock')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    commands=[];expected={};periods={};formats={};tag=0
    for fmt in Format:
        width=WIDTHS[fmt];aw=ACC_WIDTHS[fmt];lanes=64//width;tiles=n//(cols*lanes)
        one=1 if fmt<Format.F4 else to_bits(1,fmt)
        value=k if fmt<Format.F4 else to_bits(k,Format.F64 if fmt==Format.F64 else Format.F32)
        masks=[(1<<lanes)-1]*cols;aa=[one]*rows;bb=[pack([one]*lanes,width)]*cols
        payload=' '.join(f'{v:x}'for v in masks+aa+bb)
        for base in range(0,tiles,16):
            contexts=min(16,tiles-base);begin=tag;commands.append(f'T {begin}\n');tag+=1
            for t in range(k):
                for ctx in range(contexts):
                    commands.append(f'I {int(fmt)} {ctx} {int(t==0)} {tag} {(1<<rows)-1:x} {payload}\n');tag+=1
                gap=max(0,caps['feedback_issue_interval']-contexts)
                # Avoid counting a deliberate trailing idle interval after K.
                if gap and t+1<k:commands.append(f'W {gap}\n')
            commands.append(f'D\nT {tag}\n');periods[tag]=(begin,fmt);tag+=1
            for ctx in range(contexts):
                for r in range(rows):
                    for c in range(cols):
                        expected[tag]=pack([value]*lanes,aw)
                        commands.append(f'P {ctx} {r} {c} {tag}\n');tag+=1
    trace=out/'commands.txt';trace.write_text(''.join(commands))
    proc=subprocess.run([str(dest/'sim'),str(trace)],check=True,capture_output=True,text=True,timeout=1800)
    (out/'simulation.log').write_text(proc.stdout+proc.stderr)
    timers={};seen=set()
    for line in proc.stdout.splitlines():
        if line.startswith('R '):
            _,t,value,flags,*_=line.split();t=int(t)
            if int(value,16)!=expected[t]or int(flags):raise AssertionError(f'incorrect output {t}')
            seen.add(t)
        elif line.startswith('T '):
            _,t,cycles,macs,vectors=line.split();timers[int(t)]=(int(cycles),int(macs),int(vectors))
    if len(seen)!=len(expected)or len(timers)!=2*len(periods):raise AssertionError('incomplete timed computation')
    identity=workload_hash([[1]*k for _ in range(rows)],[[1]*n for _ in range(k)])
    for end,(begin,fmt)in periods.items():
        old=timers[begin];new=timers[end]
        counters=formats.setdefault(fmt,[0,0,0])
        for i in range(3):counters[i]+=new[i]-old[i]
    for fmt,(cycles,macs,vectors)in formats.items():
        lanes=64//WIDTHS[fmt]
        if macs!=rows*n*k or vectors*lanes!=macs:raise AssertionError('RTL operation counters disagree')
        directory=out/fmt.name;directory.mkdir(exist_ok=True)
        metrics=dict(format=fmt.name,m=rows,n=n,k=k,stages=stages,rows=rows,cols=cols,
            execution_path='data_ready_array_gemm',workload_sha256=identity,
            compute_cycles=cycles,elapsed_ps=cycles*core_ps,macs=macs,issued_pe_vectors=vectors,
            operations=2*macs,gops=2*macs/(cycles*core_ps*1e-12)/1e9,
            array_issue_utilization=vectors/(cycles*rows*cols),lane_utilization=1,
            mac_utilization=macs/(cycles*rows*cols*lanes),
            core_frequency_hz=1e12/core_ps,memory_frequency_hz=None,dram=False,dma_backend='data_ready',
            dram_library_sha256=None,timing_validated=timing_validated,
            time_kind='timing_validated'if timing_validated else'configured_clock_projection',
            exact_output_match=True,readout_in_compute_time=False,
            rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'],testbench_sha256=caps['testbench_sha256'],
            trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
        if timing_validated:
            metrics.update(timing_scope='array_core',timing_report_sha256=hashlib.sha256(Path(timing).read_bytes()).hexdigest(),
                timing_netlist_sha256=proof['netlist_sha256'],reference_process=proof['reference_process'])
        if area_evidence:
            metrics.update(area_metrics(metrics,area_evidence,'array'))
        write_metrics(directory,metrics);print(json.dumps(metrics))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stages',type=int,default=2)
    p.add_argument('--rows',type=int,default=8);p.add_argument('--cols',type=int,default=8)
    p.add_argument('--n',type=int,default=2048);p.add_argument('--k',type=int,default=8)
    p.add_argument('--core-ps',type=int,default=CORE_PS);p.add_argument('--out',default=str(ROOT/'artifacts/compute_all_formats'))
    p.add_argument('--build-dir');p.add_argument('--area');p.add_argument('--timing',help='authenticated extracted timing for this exact array RTL')
    a=p.parse_args();run(a.stages,a.rows,a.cols,a.n,a.k,a.core_ps,a.out,a.build_dir,a.area,a.timing)
