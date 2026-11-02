#!/usr/bin/env python3
"""Check actual controller DFF transistors against their Liberty edge/polarity.

The isolated x1/x2 cells have ideal clock/data sources and 2fF output loads.
This diagnoses controller startup; it does not qualify the complete SRAM.
"""
import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

from asap7_platform import profile
from check_sram_macro_transistor import hierarchy
from check_sram_xyce import native_models,native_devices
from prepare_xyce import BASE,LOCK,ROOT,sha
from pvt import operating_point
from sram_package import load_package
from simd_eval.architecture import ARCH


def run(args):
    package=load_package(args.sram_package,ARCH);out=args.out.resolve()
    if out.exists():raise ValueError('preserve prior transistor diagnostics')
    tools=json.loads((BASE/'capabilities.json').read_text())
    if sha(tools['executable'])!=tools['executable_sha256']or sha(LOCK)!=tools['lock_sha256']:
        raise ValueError('native simulator identity changed')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'experiment_driver.py')
    (out/'toolchain.json').write_text(json.dumps(tools,indent=2)+'\n')
    source=package['root']/(package['native']+'.sp');raw=source.read_text()
    definitions,ports=hierarchy(raw);names=['dffhqnx1_asap7_75t_r','dffhqnx2_asap7_75t_r']
    for name in names:
        if [p.lower()for p in ports[name]]!=['clk','d','qn','vdd','vss']or len(definitions[name])!=24:
            raise ValueError('unexpected actual DFF circuit')
    records=[]
    for corner in('TT','SS','FF'):
        directory=out/corner;directory.mkdir();lib=profile(corner)['liberty'];library=lib.read_text()
        # Require the documented inverting output on the rising edge. Do not
        # derive an expected value from the measured transistor waveform.
        for cell in('DFFHQNx1_ASAP7_75t_R','DFFHQNx2_ASAP7_75t_R'):
            start=library.index('cell ('+cell+')');end=library.find('\n  cell (',start+1)
            body=library[start:end if end>=0 else len(library)]
            for expression in('clocked_on : "CLK";','next_state : "!D";','function : "IQN";'):
                if expression not in body:raise ValueError('Liberty sequential expectation changed')
        point=operating_point(lib)
        if point['problems']:raise ValueError('invalid PVT')
        v=point['values']['nom_voltage'];temp=point['values']['nom_temperature']
        model=package['root']/'models'/('7nm_'+corner+'.pm')
        (directory/'models.inc').write_text(native_models(model.read_text()))
        (directory/'macro.inc').write_text(native_devices(raw))
        script=f'''Actual controller DFF edge/polarity diagnostic
.include models.inc
.include macro.inc
.param supply={v}
.options device temp={temp} tnom=25
VDD vdd 0 {{supply}}
VCLK clk 0 PULSE(0 {{supply}} .5n 10p 10p .99n 2n)
VD data 0 PWL(0 0 .8n 0 .81n {{supply}} 2.8n {{supply}} 2.81n 0 5.8n 0 5.81n {{supply}})
'''
        for i,name in enumerate(names):
            instance='Xf'+str(i);script+=f'{instance} clk data q{i} vdd 0 {name}\nC{i} q{i} 0 2f\n'
            states={'clkn':1,'clkb':0,'MH':1,'MS':0,'SH':1,'SS':0,'pd1':0,'pd2':1,
                'pd3':0,'pd4':1,'pd5':1,'pu1':1,'QN':0}
            aliases={'MH':'net15','MS':'net29','SH':'net32','SS':'net37'}if i else{}
            nodes={n.lower()for f in definitions[name]for n in f[1:5]}
            for node,value in states.items():
                node=aliases.get(node,node)
                if node.lower()not in nodes:raise ValueError('DFF initial-state topology changed')
                script+=f'.ic V({instance}:{node})='+('{supply}'if value else'0')+'\n'
        script+='''.tran 2p 8n 0 2p uic
.PRINT TRAN FORMAT=CSV FILE=voltages.csv V(clk) V(data) V(q0) V(q1) {TEMP}
.end
'''
        (directory/'test.cir').write_text(script)
        env={**os.environ,'LD_LIBRARY_PATH':str(BASE/'install/suitesparse/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')}
        timed_out=False
        with(directory/'xyce.log').open('w')as stream:
            process=subprocess.Popen([tools['executable'],'test.cir'],cwd=directory,env=env,stdout=stream,stderr=subprocess.STDOUT)
            try:code=process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:timed_out=True;process.kill();code=process.wait()
        samples=[];wave=directory/'voltages.csv'
        if wave.exists():
            with wave.open()as stream:
                for row in csv.DictReader(stream):
                    values=list(map(float,row.values()))
                    if len(values)!=6 or not all(map(math.isfinite,values)):raise ValueError('invalid DFF waveform')
                    samples.append(values)
        checks=[]
        for t,one in[(.9,True),(1.8,True),(2.8,False),(3.8,False),(4.8,True),(5.8,True),(6.8,False),(7.8,False)]:
            row=min(samples,key=lambda r:abs(r[0]-t*1e-9))if samples else None
            for i,name in enumerate(names):
                passed=bool(row)and abs(row[0]-t*1e-9)<=2e-12 and abs(row[-1]-temp)<1e-5
                if passed:passed=row[3+i]>.8*v if one else row[3+i]<.2*v
                checks.append(dict(cell=name,time_ns=t,expected_QN=int(one),sample=row,passed=passed))
        log=(directory/'xyce.log').read_text()
        clean=not re.search(r'(?i)(fatal|error:|will be ignored|time step too small|simulation aborted)',log)
        actual_temp=bool(samples)and all(abs(row[-1]-temp)<1e-5 for row in samples)
        record=dict(corner=corner,voltage_V=v,temperature_C=temp,temperature_validated=actual_temp,
            source_liberty_sha256=sha(lib),source_model_sha256=sha(model),returncode=code,timed_out=timed_out,
            waveform_rows=len(samples),checks=checks,passed=code==0 and not timed_out and clean and actual_temp and all(c['passed']for c in checks))
        (directory/'operating_point.json').write_text(json.dumps(record,indent=2)+'\n');records.append(record)
    result=dict(kind='actual_controller_dff_edge_and_polarity_diagnostic',passed=all(r['passed']for r in records),
        operating_points=records,macro_package_manifest_sha256=package['manifest_sha256'],source_schematic_sha256=sha(source),
        simulator_sha256=tools['executable_sha256'],bitcells=0,transistors=48,clock_period_ns=2,output_load_fF=2,
        expected_function='Rising CLK captures QN=!D; QN holds between rising edges',ideal_control_signals=True,
        equations_modified=False,whole_macro_characterized=False,physical_frequency_certified=False,
        evidence={str(p):sha(p)for p in out.rglob('*')if p.is_file()})
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='evidence'},indent=2))
    if not result['passed']:raise RuntimeError('actual controller DFF diagnostic failed; evidence retained')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sram-package',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--timeout',type=int,default=90);args=p.parse_args()
    if not 1<=args.timeout<=300:p.error('timeout must be within1..300seconds')
    run(args)
