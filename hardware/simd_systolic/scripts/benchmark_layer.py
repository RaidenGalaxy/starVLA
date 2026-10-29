#!/usr/bin/env python3
"""Evaluate one source layer using the same frozen GEMM simulator and clocks."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from runtime import validated_build

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
import sys as _architecture_sys
_architecture_sys.path.insert(0,str(ROOT/'python'))
from simd_eval.architecture import CORE_PS, MEMORY_PS
from simd_eval.reporting import compare
from simd_eval.area import load_area,area_metrics
from simd_eval.reporting import write_metrics


def write_comparisons(out,modes):
    baseline=json.loads((out/'BF16/metrics.json').read_text());comparisons={};table=[]
    reports={'BF16':baseline,**{mode:json.loads((out/mode/'metrics.json').read_text())for mode in modes}}
    for mode in modes:
        comparisons[mode]=compare(baseline,reports[mode])
        (out/(mode+'_comparison.json')).write_text(json.dumps(comparisons[mode],indent=2)+'\n')
    for mode,report in reports.items():
        row=dict(mode=mode,format=report['format'],elapsed_ms=report['elapsed_ps']/1e9,
          speedup=1 if mode=='BF16'else comparisons[mode]['speedup'],
          relative_l2=report.get('relative_l2'),framework_relative_l2=report.get('framework_relative_l2'),
          gmac_s=report.get('gmac_s'),dram_physical_bytes=report.get('dram_physical_bytes'),
          dma_transfer_bytes=report.get('dma_transfer_bytes'),auxiliary_vectors=report.get('dequant_operations',0),
          timing_validated=report['timing_validated'],rtl_sha256=report['rtl_sha256'])
        row.update({key:report.get(key)for key in('cell_area_um2','area_scope','area_netlist_sha256',
            'area_full_storage_capacity','area_retained_memory_bytes','area_mapping_profile','gops_per_mm2')})
        table.append(row)
    with(out/'comparison.csv').open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    result=comparisons[modes[0]]if len(modes)==1 else comparisons
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def annotate_area(out,path,modes):
    evidence=load_area(path);updates=[]
    for name in ['BF16',*modes]:
        directory=out/name;metrics=json.loads((directory/'metrics.json').read_text())
        updates.append((directory,{**metrics,**area_metrics(metrics,evidence,'gemm')}))
    # Validate every precision before modifying any report.
    for directory,metrics in updates:write_metrics(directory,metrics)


def benchmark(args):
    build=args.build_dir.resolve();guard,caps=validated_build(build)
    if caps['target']!='gemm':raise ValueError('an autonomous GEMM simulator is required')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    common=['--stages',str(caps['stages']),'--rows',str(caps['rows']),'--cols',str(caps['cols']),
        '--build-dir',str(build),'--core-ps',str(args.core_ps),'--memory-ps',str(args.memory_ps),
        '--npz',str(args.npz.resolve()),'--timeout',str(args.timeout)]
    if args.dram:common+=['--dram']
    if args.dram_config:common+=['--dram-config',str(args.dram_config.resolve())]
    modes=args.modes or[args.mode]
    if len(set(modes))!=len(modes):raise ValueError('candidate modes must be unique')
    if args.max_parallel<1:raise ValueError('parallel process limit must be positive')
    jobs=[('BF16','check_descriptor.py',['--format','BF16','--reference-backend',args.reference_backend])]+[
        (mode,'check_grouped.py',['--mode',mode,'--group-log2',str(args.group_log2),
          '--reference-backend',args.reference_backend])for mode in modes]
    pending=[];statuses={};streams=[]
    try:
        for name,script,options in jobs:
            if args.parallel and len(pending)>=args.max_parallel:
                finished=[]
                while not finished:
                    finished=[item for item in pending if item[1].poll()is not None]
                    if not finished:time.sleep(.2)
                for previous,process in finished:
                    statuses[previous]=process.wait();pending.remove((previous,process))
                if any(statuses.values()):break
            stream=(out/(name+'.log')).open('w');streams.append(stream)
            process=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script),*common,*options,'--out',str(out/name)],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            pending.append((name,process))
            if not args.parallel:
                statuses[name]=process.wait()
                pending.pop()
                if statuses[name]:break
        for name,process in pending:statuses[name]=process.wait()
    finally:
        for stream in streams:stream.close()
        (out/'job_status.json').write_text(json.dumps(statuses,indent=2)+'\n')
    if len(statuses)!=len(jobs) or any(statuses.values()):raise RuntimeError('layer evaluation failed; inspect individual logs and job_status.json')
    if args.area:annotate_area(out,args.area,modes)
    print(json.dumps(write_comparisons(out,modes)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--npz',type=Path,required=True)
    parser.add_argument('--build-dir',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--area',type=Path,help='authenticated synthesis.json for this exact complete GEMM system')
    parser.add_argument('--mode',choices=['W4A4','W4A8','W8A8','FP4','FP8'],default='W4A8');parser.add_argument('--group-log2',type=int,default=5)
    parser.add_argument('--modes',nargs='+',choices=['W4A4','W4A8','W8A8','FP4','FP8'],help='share one baseline across multiple candidate precisions')
    parser.add_argument('--core-ps',type=int,default=CORE_PS);parser.add_argument('--dram',action='store_true')
    parser.add_argument('--memory-ps',type=int,default=MEMORY_PS,help='2000 ps configures the 500 MHz memory domain')
    parser.add_argument('--reference-backend',choices=['rational','cpu'],default='rational',
        help='optional sequential host FP32 FMA oracle; grouped scale arithmetic and integer partials remain exact rational')
    parser.add_argument('--dram-config',type=Path);parser.add_argument('--timeout',type=int,default=1800)
    parser.add_argument('--parallel',action='store_true');parser.add_argument('--max-parallel',type=int,default=2)
    benchmark(parser.parse_args())
