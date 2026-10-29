#!/usr/bin/env python3
"""Ablate autonomous next-K-tile loading on the same RTL and clock domains."""
import argparse,contextlib,hashlib,json
from pathlib import Path
from check_descriptor import check as check_descriptors
from runtime import validated_build
from simd_eval.architecture import CORE_PS, MEMORY_PS


def check(build,out,dram,core_ps,memory_ps):
    guard,caps=validated_build(build)
    if caps['target']!='gemm'or not caps.get('autonomous_prefetch')or not caps.get('inline_chisel_assertions'):
        raise ValueError('native prefetch GEMM with embedded assertions required')
    out.mkdir(parents=True,exist_ok=True);reports={};jobs={}
    for enabled,name in [(False,'barrier'),(True,'overlap')]:
        directory=out/name
        with(out/(name+'.log')).open('w')as log,contextlib.redirect_stdout(log):
            check_descriptors(stages=caps['stages'],rows=caps['rows'],cols=caps['cols'],dram=dram,
                out=directory,core_ps=core_ps,memory_ps=memory_ps,build_dir=build,prefetch=enabled)
        reports[name]=json.loads((directory/'metrics.json').read_text())
        jobs[name]=json.loads((directory/'jobs.json').read_text())
    baseline,candidate=reports['barrier'],reports['overlap']
    for key in('rtl_sha256','simulator_sha256','testbench_sha256','dram_library_sha256',
               'macs','compute_cycles','verified_output_bytes','dma_read_requests','output_payload_bytes'):
        if baseline[key]!=candidate[key]:raise AssertionError('prefetch ablation changed '+key)
    for tag,job in jobs['barrier'].items():
        other=jobs['overlap'][tag]
        for key in('reference','flags','compute_cycles','memory_cycles'):
            if key=='memory_cycles':continue # overlaps deliberately change memory-domain elapsed time
            if key in job and job[key]!=other[key]:raise AssertionError('prefetch changed job '+tag+' '+key)
    actual_outputs={}
    for name in('barrier','overlap'):
        lines=(out/name/'simulation.log').read_text().splitlines()
        values=sorted(line for line in lines if line.startswith('W '))
        actual_outputs[name]=hashlib.sha256('\n'.join(values).encode()).hexdigest()
    if actual_outputs['barrier']!=actual_outputs['overlap']:
        raise AssertionError('actual output bytes changed under prefetch')
    if baseline['prefetched_tiles']or baseline['prefetch_load_bytes']:
        raise AssertionError('disabled prefetch still loaded a future tile')
    if candidate['prefetch_load_bytes_while_core_active']<=0:
        raise AssertionError('no actual SRAM input transfer overlapped a running core')
    trace_hash=lambda name:hashlib.sha256((out/name/'commands.txt').read_bytes()).hexdigest()
    report=dict(passed=True,scope='autonomous_gemm_prefetch_ablation',stages=caps['stages'],rows=caps['rows'],cols=caps['cols'],
        descriptors=candidate['descriptors'],all_formats=True,initial_c=True,w4a8=True,w4a16=True,
        organization='current complete tile in wave FIFO; next K tile in A/B SRAM',
        group_boundary_policy='finish current group before prefetching across a boundary',dram=dram,
        core_period_ps=core_ps,memory_period_ps=memory_ps,noninteger_clock_ratio=core_ps%memory_ps!=0 and memory_ps%core_ps!=0,
        barrier_elapsed_ps=baseline['elapsed_ps'],overlap_elapsed_ps=candidate['elapsed_ps'],
        scheduling_speedup=baseline['elapsed_ps']/candidate['elapsed_ps'],
        prefetched_tiles=candidate['prefetched_tiles'],prefetch_load_bytes=candidate['prefetch_load_bytes'],
        overlapped_sram_load_bytes=candidate['prefetch_load_bytes_while_core_active'],
        identical_outputs_compute_cycles_and_logical_traffic=True,
        actual_output_sha256=actual_outputs['barrier'],
        rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'],
        barrier_trace_sha256=trace_hash('barrier'),overlap_trace_sha256=trace_hash('overlap'),
        time_kind='configured_clock_projection',timing_validated=False)
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--dram',action='store_true')
    parser.add_argument('--core-ps',type=int,default=CORE_PS);parser.add_argument('--memory-ps',type=int,default=MEMORY_PS)
    args=parser.parse_args();check(args.build_dir.resolve(),args.out.resolve(),args.dram,args.core_ps,args.memory_ps)
