#!/usr/bin/env python3
"""Exercise actual 6T cells and the macro's complete I/O peripheral circuit.

Eight explicit 10fF bitline pairs replace the array loads; ideal control signals
replace the row decoder/controller. No array timing or extracted-layout claim.
--array retains one native 1024-cell column and its dummy/peripheral devices;
the other 63 columns, real row drivers and layout RC are still absent.
"""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

from asap7_platform import profile
from check_sram_macro_transistor import hierarchy
from check_sram_xyce import native_models,native_devices
from prepare_xyce import BASE,LOCK,ROOT,sha
from pvt import operating_point
from sram_package import load_package
from simd_eval.architecture import ARCH


def run(args):
    package=load_package(args.sram_package,ARCH)
    tools=json.loads((BASE/'capabilities.json').read_text())
    if sha(tools['executable'])!=tools['executable_sha256']or sha(LOCK)!=tools['lock_sha256']:
        raise ValueError('native toolchain identity changed')
    out=args.out.resolve()
    if out.exists():raise ValueError('preserve prior read-path diagnostics')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'experiment_driver.py')
    (out/'toolchain.json').write_text(json.dumps(tools,indent=2)+'\n')
    source=package['root']/(package['native']+'.sp');text=source.read_text()
    definitions,ports=hierarchy(text);io='iocolgrp_sram_6t122_v2'
    def count(name):
        return sum(1 if f[0].lower().startswith('m')else count(f[-1].lower())if f[0].lower().startswith('x')else 0 for f in definitions[name])
    native='colgrp_sram_6t122'if args.array else io
    cell_paths=[];dummy_nodes=[]
    def visit(name,path):
        if name=='sram_cell_6t_122':cell_paths.append(path)
        if name=='dummy_sram_6t122':dummy_nodes.append(path+':QB')
        if name=='dummy_topbot_v1':dummy_nodes.append(path+':6')
        for fields in definitions[name]:
            if fields[0].lower().startswith('x'):visit(fields[-1].lower(),path+':'+fields[0])
    if args.array:
        visit(native,'Xcol')
        if len(cell_paths)!=1024:raise ValueError('native column cell organization changed')
        transistors=count(native)
    else:cell_paths=['Xcell'];transistors=count(io)+count('sram_cell_6t_122')
    selected_cell=cell_paths[0];results=[]
    for corner in(args.corner,)if args.corner else('TT','SS','FF'):
        directory=out/corner;directory.mkdir()
        point=operating_point(profile(corner)['liberty'])
        if point['problems']:raise ValueError('invalid corner operating point')
        voltage=point['values']['nom_voltage'];temperature=point['values']['nom_temperature']
        model=package['root']/'models'/('7nm_'+corner+'.pm')
        (directory/'models.inc').write_text(native_models(model.read_text()))
        (directory/'macro.inc').write_text(native_devices(text))
        mapping={'wrenan':'wn','wrena':'we','sae':'sae','saprechn':'sap','oeb_out':'0',
            'oe_out':'vdd','d':'data','q':'qo','blprechtn':'bpre','blprechbn':'0','vdd':'vdd','vss':'0'}
        for side in('t','b'):
            for i in range(4):
                column_selected=side=='t'and i==0
                mapping[f'ysel{side}[{i}]']='vdd'if column_selected else'0'
                mapping[f'ysel{side}n[{i}]']='0'if column_selected else'vdd'
                mapping[f'bl{side}[{i}]']=f'bl{side}{i}'
                mapping[f'bl{side}n[{i}]']=f'bl{side}n{i}'
        if args.array:
            mapping={key:value for key,value in mapping.items()if not key.startswith(('blt[','bltn[','blb[','blbn['))}
            mapping.update({f'wl{side}[{i}]':'wl'if side=='t'and i==0 else'0'
                for side in('t','b')for i in range(128)})
        if set(mapping)!=set(p.lower()for p in ports[native]):raise ValueError('native peripheral port map changed')
        script=f'''Actual SRAM native read-path diagnostic, ideal control signals
.include models.inc
.include macro.inc
.param supply={voltage}
.options device temp={temperature} tnom=25
VDD vdd 0 {{supply}}
VD data 0 PWL(0 {{supply}} 4n {{supply}} 4.1n 0)
VWL wl 0 PWL(0 0 .3n 0 .31n {{supply}} .7n {{supply}} .71n 0
+ 2.3n 0 2.31n {{supply}} 2.7n {{supply}} 2.71n 0
+ 4.3n 0 4.31n {{supply}} 4.7n {{supply}} 4.71n 0
+ 6.3n 0 6.31n {{supply}} 6.7n {{supply}} 6.71n 0)
VWN wn 0 PWL(0 {{supply}} .2n {{supply}} .21n 0 .81n 0 .82n {{supply}}
+ 4.2n {{supply}} 4.21n 0 4.81n 0 4.82n {{supply}})
VWE we 0 PWL(0 0 .21n 0 .22n {{supply}} .8n {{supply}} .81n 0
+ 4.21n 0 4.22n {{supply}} 4.8n {{supply}} 4.81n 0)
VPRE bpre 0 PWL(0 0 .1n 0 .11n {{supply}}
+ 2n {{supply}} 2.01n 0 2.1n 0 2.11n {{supply}}
+ 4n {{supply}} 4.01n 0 4.1n 0 4.11n {{supply}}
+ 6n {{supply}} 6.01n 0 6.1n 0 6.11n {{supply}})
VSAP sap 0 PWL(0 0 2.49n 0 2.5n {{supply}} 2.9n {{supply}} 2.91n 0
+ 6.49n 0 6.5n {{supply}} 6.9n {{supply}} 6.91n 0)
VSAE sae 0 PWL(0 0 2.51n 0 2.52n {{supply}} 2.85n {{supply}} 2.86n 0
+ 6.51n 0 6.52n {{supply}} 6.85n {{supply}} 6.86n 0)
'''
        if args.array:script+=f"Xcol {' '.join(mapping[p.lower()]for p in ports[native])} {native}\n"
        else:script+=f"Xcell wl bltn0 blt0 vdd 0 sram_cell_6t_122\nXio {' '.join(mapping[p.lower()]for p in ports[io])} {io}\n"
        script+='CQ qo 0 2f\n'
        # Deterministic power-on choices initialize actual storage nodes only;
        # no ideal source holds them after transient integration starts.
        for i,path in enumerate(cell_paths):
            q='0'if i%2==0 else'{supply}';qb='{supply}'if i%2==0 else'0'
            script+=f'.ic V({path}:Q)={q} V({path}:QB)={qb}\n'
        script+=''.join(f'.ic V({path})={{supply}}\n'for path in dummy_nodes)
        for key,value in mapping.items():
            if key.startswith('blt[')or key.startswith('bltn[')or key.startswith('blb[')or key.startswith('blbn['):
                script+=f'C{value} {value} 0 10f\n.ic V({value})={{supply}}\n'
        bl='Xcol:blt[0]'if args.array else'blt0';bln='Xcol:bltn[0]'if args.array else'bltn0'
        script+=f'''.tran {min(5,args.max_step_ps)}p 8n 0 {args.max_step_ps}p uic
.PRINT TRAN FORMAT=CSV FILE=voltages.csv V({selected_cell}:Q) V({selected_cell}:QB) V(qo) V({bl}) V({bln}) {{TEMP}}
.end
'''
        if args.array:
            variables=' '.join(f'V({path}:Q) V({path}:QB)'for path in cell_paths)+' {TEMP}'
            chunks=textwrap.wrap(variables,width=120,break_long_words=False,break_on_hyphens=False)
            script=script.replace('.end\n','.PRINT TRAN FORMAT=CSV FILE=retention.csv\n'+
                '\n'.join('+ '+line for line in chunks)+'\n.end\n')
        (directory/'test.cir').write_text(script)
        env={**os.environ,'LD_LIBRARY_PATH':str(BASE/'install/suitesparse/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')}
        timed_out=False
        with(directory/'xyce.log').open('w')as stream:
            process=subprocess.Popen([tools['executable'],'test.cir'],cwd=directory,env=env,stdout=stream,stderr=subprocess.STDOUT)
            (directory/'running.json').write_text(json.dumps(dict(pid=process.pid,timeout_seconds=args.timeout))+'\n')
            try:code=process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:timed_out=True;process.kill();code=process.wait()
        samples=[];wave=directory/'voltages.csv'
        if wave.exists():
            with wave.open()as stream:
                for row in csv.DictReader(stream):
                    values=list(map(float,row.values()))
                    if len(values)!=7 or not all(map(math.isfinite,values)):raise ValueError('invalid read-path waveform')
                    samples.append(values)
        checks=[]
        for name,t,one,read in [('write_one',.65,True,False),('hold_one',1.5,True,False),
                ('read_one',3.2,True,True),('write_zero',4.65,False,False),('hold_zero',5.5,False,False),('read_zero',7.2,False,True)]:
            row=min(samples,key=lambda r:abs(r[0]-t*1e-9))if samples else None
            passed=bool(row)and abs(row[0]-t*1e-9)<=args.max_step_ps*1e-12
            if passed:
                _,q,qb,qo,bl,bln,actual_temp=row
                passed=abs(actual_temp-temperature)<1e-5 and(q>.8*voltage and qb<.2*voltage if one else qb>.8*voltage and q<.2*voltage)
                if read:passed=passed and(qo>.8*voltage if one else qo<.2*voltage)
            checks.append(dict(name=name,time_ns=t,sample=row,sample_tolerance_ps=args.max_step_ps,passed=passed))
        retention_checks=[];retention_rows=0
        if args.array:
            closest=[None]*len(checks);retention_wave=directory/'retention.csv'
            if retention_wave.exists():
                with retention_wave.open()as stream:
                    for row in csv.DictReader(stream):
                        values=list(map(float,row.values()));retention_rows+=1
                        if len(values)!=2*len(cell_paths)+2 or not all(map(math.isfinite,values)):
                            raise ValueError('invalid whole-column retention waveform')
                        if abs(values[-1]-temperature)>=1e-5:raise ValueError('retention waveform has the wrong actual temperature')
                        for i,check in enumerate(checks):
                            t=check['time_ns']*1e-9
                            if closest[i]is None or abs(values[0]-t)<abs(closest[i][0]-t):closest[i]=values
            for check,row in zip(checks,closest):
                stable=[]
                if row:
                    for i,path in enumerate(cell_paths[1:],1):
                        q,qb=row[1+2*i:3+2*i]
                        high,low=(q,qb)if i%2 else(qb,q)
                        if not(high>.8*voltage and low<.2*voltage):stable.append(path)
                passed=bool(row)and abs(row[0]-check['time_ns']*1e-9)<=args.max_step_ps*1e-12 and not stable
                retention_checks.append(dict(name=check['name'],time_ns=check['time_ns'],
                    sampled_time_ns=row[0]*1e9 if row else None,unselected_cells=len(cell_paths)-1,
                    failed_cells=stable,passed=passed))
        log=(directory/'xyce.log').read_text()
        clean=not any(s in log.lower()for s in('fatal','error:','will be ignored','time step too small','simulation aborted'))
        temperature_valid=bool(samples)and all(abs(r[-1]-temperature)<1e-5 for r in samples)
        results.append(dict(corner=corner,voltage_V=voltage,temperature_C=temperature,temperature_validated=temperature_valid,
            source_model_sha256=sha(model),returncode=code,timed_out=timed_out,waveform_rows=len(samples),checks=checks,
            retention_checks=retention_checks,retention_waveform_rows=retention_rows,
            passed=code==0 and not timed_out and clean and temperature_valid and all(c['passed']for c in checks+retention_checks)))
    result=dict(kind='native_1024_cell_column_with_ideal_controls'if args.array else'one_6t_cell_with_actual_complete_macro_io_peripherals',passed=all(r['passed']for r in results),
        operating_points=results,macro_package_manifest_sha256=package['manifest_sha256'],source_schematic_sha256=sha(source),
        simulator_sha256=tools['executable_sha256'],bitcells=len(cell_paths),transistors=transistors,
        external_bitline_pairs=0 if args.array else 8,bitline_capacitance_fF=None if args.array else 10,
        actual_single_bit_column_loading=args.array,initialized_dummy_nodes=dummy_nodes,
        initialized_storage_pattern='alternating 0/1; initial conditions only',
        output_capacitance_fF=2,operation_spacing_ns=2,
        maximum_integration_step_ps=args.max_step_ps,
        ideal_control_signals=True,actual_row_controller=False,complete_array_loading=False,
        equations_modified=False,schematic_only=True,whole_macro_characterized=False,physical_frequency_certified=False,
        evidence={str(p):sha(p)for p in out.rglob('*')if p.is_file()})
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='evidence'},indent=2))
    if not result['passed']:raise RuntimeError('actual SRAM read-path diagnostic failed; evidence retained')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sram-package',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--array',action='store_true',help='retain the actual 1024-cell native column, ideal control sources')
    p.add_argument('--corner',choices=['TT','SS','FF'],help='diagnostic single corner; default checks all three')
    p.add_argument('--max-step-ps',type=int,default=1,help='maximum integration step; checks retain the actual sample time')
    p.add_argument('--timeout',type=int,default=120);args=p.parse_args()
    if not 1<=args.timeout<=600:p.error('timeout must be within1..600 seconds')
    if not 1<=args.max_step_ps<=20:p.error('maximum step must be within1..20ps')
    run(args)
