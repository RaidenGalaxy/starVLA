#!/usr/bin/env python3
"""Export numerical/performance comparisons from authenticated experiment reports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from simd_eval.reporting import compare


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=args.directory.resolve();modes=['BF16','W4A4','W4A8','W8A8','FP4','FP8']
    paths={mode:root/mode/'metrics.json'for mode in modes}
    paths={mode:path for mode,path in paths.items()if path.is_file()}
    if 'BF16'not in paths or len(paths)<2:raise ValueError('a BF16 baseline and at least one candidate are required')
    reports={mode:json.loads(path.read_text())for mode,path in paths.items()};baseline=reports['BF16']
    ratios={mode:1 if mode=='BF16'else compare(baseline,report)['speedup']for mode,report in reports.items()}
    metadata=json.loads(args.workload_metadata.read_text())if args.workload_metadata else{}
    if metadata and any(report.get('source_npz_sha256')!=metadata['npz_sha256']for report in reports.values()):
        raise ValueError('workload provenance differs from the evaluated input')
    modes=list(reports);colors=dict(zip(['BF16','W4A4','W4A8','W8A8','FP4','FP8'],
        ['#64748b','#2563eb','#38bdf8','#14b8a6','#8b5cf6','#db2777']))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none',
        'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    figure,axes=plt.subplots(1,3,figsize=(12,4.1),gridspec_kw={'width_ratios':[1,1,1.2]})
    x=list(range(len(modes)));palette=[colors[mode]for mode in modes]
    for axis,values,label in[(axes[0],[reports[m]['elapsed_ps']/1e9 for m in modes],'End-to-end time (ms)'),
                             (axes[1],[reports[m]['dram_physical_bytes']/1e6 for m in modes],'Actual DRAM transfers (MB)')]:
        bars=axis.bar(x,values,color=palette,width=0.72)
        axis.set_xticks(x,modes,rotation=35,ha='right');axis.set_ylabel(label)
        axis.set_ylim(0,max(values)*1.2);axis.grid(axis='y',alpha=.18);axis.set_axisbelow(True)
        for bar,value in zip(bars,values):axis.text(bar.get_x()+bar.get_width()/2,value+max(values)*.025,f'{value:.3f}',ha='center',fontsize=8)
    error_key='framework_relative_l2'if all('framework_relative_l2'in report for report in reports.values())else'relative_l2'
    for mode in modes:
        error=reports[mode][error_key]
        if error is None:raise ValueError('nonfinite or unavailable error metric')
        axes[2].scatter(error*100,ratios[mode],s=65,color=colors[mode],zorder=3)
        axes[2].annotate(mode,(error*100,ratios[mode]),xytext=(5,5),textcoords='offset points',fontsize=8)
    axes[2].axhline(1,color='#64748b',linestyle='--',linewidth=.8)
    axes[2].set_xlabel('Relative L2 error vs '+('framework'if error_key.startswith('framework')else'mathematical GEMM')+' (%)')
    axes[2].set_ylabel('Speedup over BF16');axes[2].grid(alpha=.18)
    axes[2].margins(x=.2,y=.22)
    title=f'{baseline["rows"]}x{baseline["cols"]} F{baseline["stages"]} | {baseline["core_frequency_hz"]/1e6:g} MHz | {baseline["m"]} x {baseline["n"]} x {baseline["k"]}'
    if baseline['stages']!=2:title=title.replace(f' F{baseline["stages"]} ',f' HF{baseline["stages"]} ')
    if metadata:title+=' | '+metadata['phase']+(' crop'if metadata['cropped']else' full Linear')
    figure.suptitle(title,fontsize=11,y=.98)
    validated=all(report['timing_validated']for report in reports.values())
    note='All precisions use the same RTL implementation. '+('Extracted timing validated.'if validated else'Configured-clock RTL projection; full PE/array timing not certified.')
    figure.text(.02,.015,note,fontsize=8,color='#475569')
    figure.tight_layout(rect=(0,.055,1,.94))
    out=args.out.resolve()if args.out else root/'figures';out.mkdir(parents=True,exist_ok=True)
    outputs={}
    for extension in('svg','png','pdf'):
        path=out/('comparison.'+extension);figure.savefig(path,dpi=180)
        outputs[extension]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    plt.close(figure)
    if all(report.get('sequencer_cycle_accounting_exact')for report in reports.values()):
        categories=list(baseline['sequencer_cycle_breakdown'])
        if any(list(report['sequencer_cycle_breakdown'])!=categories for report in reports.values()):
            raise ValueError('sequencer phase categories differ')
        phase,phase_axes=plt.subplots(1,2,figsize=(12,5.1))
        elapsed=[reports[mode]['elapsed_ps']/1e6 for mode in modes]
        values=[]
        for category in categories:
            values.append([reports[mode]['sequencer_cycle_breakdown'][category]*1e6/reports[mode]['memory_frequency_hz']for mode in modes])
        outside=[elapsed[i]-sum(row[i]for row in values)for i in x]
        if min(outside)<-1e-6:raise ValueError('sequencer phases exceed the end-to-end elapsed time')
        categories+=['completion_idle_external_drain'];values.append([max(0,v)for v in outside])
        phase_colors=plt.get_cmap('tab10').colors
        labels={'address_pack_control':'Address / packing / control','initial_c_seed':'Initial C load / seed',
            'dram_request_wait':'DRAM request handshake','dram_response_wait':'DRAM response wait',
            'tile_issue_wait':'Tile command handshake','tile_completion_wait':'Core / tile completion',
            'result_readback':'Result readback / CDC','auxiliary_scale_convert':'Scale / convert / auxiliary',
            'output_write':'Output DMA / response','completion_idle_external_drain':'Completion / idle / external drain'}
        for axis,percent in zip(phase_axes,[False,True]):
            bottoms=[0.0]*len(modes)
            for i,(category,row)in enumerate(zip(categories,values)):
                heights=[100*value/total if percent else value for value,total in zip(row,elapsed)]
                axis.bar(x,heights,bottom=bottoms,color=phase_colors[i%10],label=labels.get(category,category),width=.72)
                bottoms=[a+b for a,b in zip(bottoms,heights)]
            axis.set_xticks(x,modes,rotation=35,ha='right');axis.grid(axis='y',alpha=.15);axis.set_axisbelow(True)
            axis.set_ylabel('End-to-end time (%)'if percent else'End-to-end time (µs)')
            if percent:axis.set_ylim(0,100)
        handles,legend=phase_axes[0].get_legend_handles_labels()
        phase.legend(handles,legend,loc='lower center',ncol=5,fontsize=7,frameon=False,bbox_to_anchor=(.5,.025))
        phase.suptitle(title+' | exclusive sequencer phases',fontsize=11)
        phase.text(.02,.012,note,fontsize=7,color='#475569')
        phase.tight_layout(rect=(0,.19,1,.94))
        for extension in('svg','png','pdf'):
            path=out/('phases.'+extension);phase.savefig(path,dpi=180)
            outputs['phases_'+extension]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        plt.close(phase)
    record=dict(kind='layer_comparison_figure',rtl_sha256=baseline['rtl_sha256'],timing_validated=validated,
        source_reports={mode:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())for mode,path in paths.items()},
        workload_metadata=str(args.workload_metadata)if metadata else None,matplotlib_version=matplotlib.__version__,outputs=outputs)
    (out/'figure.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--workload-metadata',type=Path);parser.add_argument('--out',type=Path)
    plot(parser.parse_args())
