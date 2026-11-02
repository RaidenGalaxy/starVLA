#!/usr/bin/env python3
"""Check the actual SRAM schematic using native Xyce BSIM-CMG107.

This bounded schematic experiment retains all device equations and ASAP7
parameters. It supplies no layout parasitics or physical qualification.
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
from check_sram_bitcell import cell_port,deck
from check_sram_macro_transistor import hierarchy
from prepare_xyce import BASE,LOCK,ROOT,sha
from pvt import operating_point
from sram_package import load_package
from simd_eval.architecture import ARCH


def native_models(text):
    text,count=re.subn(r'(?mi)^(\.model[ \t]+\w+[ \t]+(?:nmos|pmos)[ \t]+level[ \t]*=[ \t]*)72[ \t]*$',r'\g<1>107',text)
    if count!=8:raise ValueError('expected eight BSIM-CMG107 model declarations')
    text,count=re.subn(r'(?i)\bversion\s*=\s*107\b','',text)
    if count!=8:raise ValueError('expected exactly the version107 selector')
    return text


def native_devices(text):
    # W is a source annotation; the native CMG model uses L and NFIN.
    lines=[];count=0
    for line in text.splitlines():
        if line[:1].lower()=='m':
            if set(k.lower()for k in re.findall(r'(\w+)\s*=',line))!={'l','w','nfin'}:
                raise ValueError('unexpected MOS geometry')
            line=re.sub(r'(?i)\s+w\s*=\s*\S+','',line);count+=1
        lines.append(line)
    if not count:raise ValueError('no native transistors')
    return '\n'.join(lines)+'\n'


def output_directives(script,vectors):
    script=script[:script.lower().index('.control')]
    script=script.replace('.options tnom=25','.options device tnom=25')
    script=re.sub(r'(?mi)^\.temp\s+([^\n]+)$',r'.options device temp=\1',script)
    script=re.sub(r'(?i)xcell\.', 'xcell:',script)
    # Xyce's documented native voltage switch has a smooth transition. Its
    # narrow 2mV band is an explicit ideal peripheral stimulus assumption.
    script=re.sub(r'(?i)SW\(RON=10 ROFF=1e15 VT=([\d.eE+-]+) VH=0\)',
        lambda m:f'VSWITCH(RON=10 ROFF=1e15 VON={float(m[1])+.001} VOFF={float(m[1])-.001})',script)
    return script+'\n.PRINT TRAN FORMAT=CSV FILE=voltages.csv '+vectors+' {TEMP}\n.end\n'


def run(args):
    tools=json.loads((BASE/'capabilities.json').read_text())
    executable=Path(tools['executable'])
    if sha(executable)!=tools['executable_sha256']or sha(LOCK)!=tools['lock_sha256']:
        raise ValueError('native simulator identity changed')
    package=load_package(args.sram_package,ARCH);out=args.out.resolve()
    if out.exists():raise ValueError('preserve previous native simulation evidence')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'experiment_driver.py')
    (out/'toolchain.json').write_text(json.dumps(tools,indent=2)+'\n')
    source=package['root']/(package['native']+'.sp');source_text=source.read_text()
    records=[]
    for corner in(args.corner,)if args.kind=='column'else('TT','SS','FF'):
        directory=out/corner;directory.mkdir()
        point=operating_point(profile(corner)['liberty'])
        if point['problems']:raise ValueError('invalid standard-cell operating point')
        v=point['values']['nom_voltage'];temperature=point['values']['nom_temperature']
        model=package['root']/'models'/('7nm_'+corner+'.pm')
        (directory/'models.inc').write_text(native_models(model.read_text()))
        if args.kind=='cell':
            original,_,devices=cell_port(source_text)
            (directory/'cell.inc').write_text(native_devices(original))
            script=output_directives(deck(v,temperature),'V(Xcell:Q) V(Xcell:QB) V(BL) V(BLN)')
            script=script.replace('.tran 1p 8n uic','.tran 1p 8n 0 1p uic')
            bitcells=1;transistors=6
        else:
            definitions,ports=hierarchy(source_text);top=package['native'].lower()
            stacked='stacked_colgrp_x256x32x1'
            children=[f for f in definitions[stacked]if f[0].lower().startswith('x')]
            if len(children)!=32:raise ValueError('unexpected column organization')
            definitions[stacked]=[children[0]]
            definitions[top]=[f for f in definitions[top]if f[0].lower()!='xdata_bottom']
            import textwrap
            emitted=[];paths=[];dummy_drains=[];controller_initial={};transistors=0
            def visit(name,path):
                nonlocal transistors
                if name=='sram_cell_6t_122':paths.append(path)
                if name=='dummy_sram_6t122':dummy_drains.append(path+':QB')
                if name=='dummy_topbot_v1':dummy_drains.append(path+':6')
                if name in('dffhqnx1_asap7_75t_r','dffhqnx2_asap7_75t_r')and args.initialize_controller_latches:
                    states={'clkn':1,'clkb':0,'MH':1,'MS':0,'SH':1,'SS':0,
                        'pd1':0,'pd2':1,'pd3':0,'pd4':1,'pd5':1,'pu1':1,'QN':0}
                    aliases={'MH':'net15','MS':'net29','SH':'net32','SS':'net37'}if'x2_'in name else{}
                    nodes={n.lower()for f in definitions[name]if f[0].lower().startswith('m')for n in f[1:5]}
                    for node,value in states.items():
                        node=aliases.get(node,node)
                        if node.lower()not in nodes:raise ValueError('unexpected controller latch topology')
                        controller_initial[path+':'+node]=value
                for fields in definitions[name]:
                    if fields[0].lower().startswith('m'):transistors+=1
                    elif fields[0].lower().startswith('x'):visit(fields[-1].lower(),path+':'+fields[0])
            visit(top,'Xmacro');bitcells=len(paths)
            if bitcells!=1024:raise ValueError('reduced column is not 1024 cells')
            for name,body in definitions.items():
                for fields in [['.SUBCKT',name,*ports[name]],*body,['.ENDS',name]]:
                    chunks=textwrap.wrap(' '.join(fields),width=120,break_long_words=False,break_on_hyphens=False)
                    emitted.append(chunks[0]);emitted.extend('+ '+part for part in chunks[1:])
            (directory/'macro.inc').write_text(native_devices('\n'.join(emitted)))
            mapping={'vdd':'vdd','vss':'0','clk':'clk','ce_n':'0','we_n':'we','oe_n':'0'}
            mapping.update({f'sdel[{i}]':'0'for i in range(4)})
            mapping.update({f'a[{i}]':'0'for i in range(10)})
            mapping.update({f'd[{i}]':'data'for i in range(64)})
            mapping.update({f'q[{i}]':'q0'if i==0 else'0'for i in range(64)})
            script=f'''One real SRAM column, controller and peripheral circuits, loads from other columns absent
.include models.inc
.include macro.inc
.options device temp={temperature}
.param supply={v}
.options device tnom=25
VDD vdd 0 {{supply}}
VCLK clk 0 PULSE(0 {{supply}} .5n 10p 10p .99n 2n)
VWE we 0 PWL(0 0 2n 0 2.01n {{supply}} 4n {{supply}} 4.01n 0 6n 0 6.01n {{supply}})
VDATA data 0 PWL(0 {{supply}} 4n {{supply}} 4.01n 0)
Xmacro {' '.join(mapping[p.lower()]for p in ports[top])} {top}
CQ0 q0 0 2f
'''
            script+=''.join(f'.ic V({p}:Q)=0 V({p}:QB)={{supply}}\n'for p in paths)
            if args.initialize_dummy_drains:
                script+=''.join(f'.ic V({p})={{supply}}\n'for p in dummy_drains)
            script+=''.join(f'.ic V({p})='+('{supply}'if value else'0')+'\n'for p,value in controller_initial.items())
            script+=f'.tran 5p 8n 0 5p {"uic"if args.startup=="uic"else""}\n.PRINT TRAN FORMAT=CSV FILE=voltages.csv V(q0) {{TEMP}}\n.end\n'
        if args.linear_solver!='default':
            script=script.replace('.PRINT TRAN',f'.options linsol type={args.linear_solver.upper()}\n.PRINT TRAN')
        (directory/'test.cir').write_text(script)
        env={**os.environ,'LD_LIBRARY_PATH':str(BASE/'install/suitesparse/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')}
        log=directory/'xyce.log';timed_out=False
        with log.open('w')as stream:
            process=subprocess.Popen([str(executable),'test.cir'],cwd=directory,env=env,
                stdout=stream,stderr=subprocess.STDOUT)
            (directory/'running.json').write_text(json.dumps(dict(pid=process.pid,timeout_seconds=args.timeout))+'\n')
            try:code=process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:timed_out=True;process.kill();code=process.wait()
        samples=[];temperatures=[];wave=directory/'voltages.csv'
        if wave.exists():
            with wave.open()as stream:
                for row in csv.DictReader(stream):
                    if 'TIME'not in row:raise ValueError('native output has no TIME column')
                    temp_keys=[key for key in row if key.strip('{} ').upper()=='TEMP']
                    if len(temp_keys)!=1:raise ValueError('native output has no unique temperature observation')
                    temp_key=temp_keys[0];temperatures.append(float(row[temp_key]))
                    values=[float(row['TIME']),*[float(val)for key,val in row.items()if key not in('TIME','Index',temp_key)]]
                    if not all(map(math.isfinite,values)):raise ValueError('nonfinite native waveform')
                    samples.append(values)
        checks=[]
        points=[('read_one',3.9,True,False),('read_zero',7.9,False,False)]if args.kind=='column'else[
            ('write_one',.65,True,False),('hold_one',1.5,True,False),('read_one',2.6,True,True),
            ('retain_one',3.2,True,False),('write_zero',4.65,False,False),('hold_zero',5.5,False,False),
            ('read_zero',6.6,False,True),('retain_zero',7.2,False,False)]
        for name,t,one,read in points:
            row=min(samples,key=lambda r:abs(r[0]-t*1e-9))if samples else None
            passed=row is not None and abs(row[0]-t*1e-9)<=5e-12
            if passed and args.kind=='column':passed=row[1]>.8*v if one else row[1]<.2*v
            elif passed:
                _,q,qb,bl,bln=row
                passed=(q-qb if one else qb-q)>.2*v
                passed=passed and ((bl-bln if one else bln-bl)>.1*v if read else(q>.8*v and qb<.2*v if one else qb>.8*v and q<.2*v))
            checks.append(dict(name=name,time_ns=t,sample=row,passed=passed))
        log_text=log.read_text()
        clean=not re.search(r'(?i)(?:error:|fatal|simulation aborted|time step too small|unrecognized|unknown parameter|will be ignored)',log_text)
        temperature_validated=bool(temperatures)and all(math.isfinite(t)and abs(t-temperature)<1e-5 for t in temperatures)
        records.append(dict(corner=corner,voltage_V=v,temperature_C=temperature,bitcells=bitcells,
            transistors=transistors,source_model_sha256=sha(model),returncode=code,timed_out=timed_out,
            temperature_observed_C=sorted(set(temperatures)),temperature_validated=temperature_validated,
            warnings_present='warning'in log_text.lower(),
            model_boundary_warning='EOTACC value'in log_text,
            waveform_rows=len(samples),checks=checks,
            passed=code==0 and not timed_out and clean and temperature_validated and all(c['passed']for c in checks)))
    result=dict(kind='native_xyce_actual_6t_cell'if args.kind=='cell'else'native_xyce_reduced_column',
        passed=all(r['passed']for r in records),operating_points=records,
        source_schematic_sha256=sha(source),macro_package_manifest_sha256=package['manifest_sha256'],
        simulator_sha256=tools['executable_sha256'],native_model_level=107,equations_modified=False,
        linear_solver=args.linear_solver,
        initialized_dummy_drains=dummy_drains if args.kind=='column'and args.initialize_dummy_drains else [],
        controller_latch_initial_conditions=controller_initial if args.kind=='column'else{},
        model_selectors_changed='HSPICE level72/version107 to native Xyce level107',
        W_annotation_removed=True,L_and_NFIN_unchanged=True,operation_spacing_ns=2,
        ideal_peripheral_drivers=args.kind=='cell',switch_transition_band_mV=2 if args.kind=='cell'else None,
        schematic_only=True,layout_parasitics_included=False,whole_macro_characterized=False,
        physical_frequency_certified=False,full_macro_drc_lvs_verified=False,
        evidence={str(p):sha(p)for p in out.rglob('*')if p.is_file()})
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='evidence'},indent=2))
    if not result['passed']:raise RuntimeError('native SRAM diagnostic failed; evidence retained')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sram-package',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--kind',choices=['cell','column'],default='cell');p.add_argument('--corner',choices=['TT','SS','FF'],default='TT')
    p.add_argument('--startup',choices=['uic','operating_point'],default='uic')
    p.add_argument('--linear-solver',choices=['default','klu2','basker'],default='default')
    p.add_argument('--initialize-dummy-drains',action='store_true',help='choose a VDD power-on state for the actual floating dummy diffusion nodes; no added circuit elements')
    p.add_argument('--initialize-controller-latches',action='store_true',help='choose consistent zero stored states for the actual controller DFFs; no circuit or device equation changes')
    p.add_argument('--timeout',type=int,default=120);args=p.parse_args()
    if not 1<=args.timeout<=3600:p.error('timeout must be within 1..3600 seconds')
    run(args)
