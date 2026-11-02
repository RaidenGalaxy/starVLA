#!/usr/bin/env python3
"""Bounded full-1024x64 schematic transistor experiment at a 2ns clock.

All array cells and real peripheral/control transistors remain instantiated.
The experiment initializes unspecified power-on storage to one valid pattern,
then writes/reads both polarities at address zero. No layout parasitics are
invented. A timeout, solver error or wrong output remains a failed diagnostic.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

from asap7_platform import profile
from check_sram_bitcell import model_port
from prepare_analog import prepare,sha
from pvt import operating_point
from sram_package import load_package
from simd_eval.architecture import ARCH


def hierarchy(text):
    logical=[]
    for line in text.splitlines():
        if line.lstrip().startswith('+'):logical[-1]+=' '+line.lstrip()[1:]
        else:logical.append(line)
    definitions={};ports={};current=None
    for line in logical:
        fields=line.split()
        if not fields or fields[0].startswith('*'):continue
        if fields[0].lower()=='.subckt':
            current=fields[1].lower();ports[current]=fields[2:];definitions[current]=[]
        elif fields[0].lower()=='.ends':current=None
        elif current:definitions[current].append(fields)
    return definitions,ports


def run(args):
    package=load_package(args.sram_package,ARCH)
    if args.resistance_formulation=='voltage':
        from prepare_resistance_model import prepare_voltage_resistance
        tools=prepare_voltage_resistance()
    elif args.resistance_formulation=='zero_sheet':
        from prepare_zero_sheet_model import prepare_zero_sheet
        tools=prepare_zero_sheet()
    else:tools=prepare()
    out=args.out.resolve()
    if out.exists():raise ValueError('preserve prior schematic experiment')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'experiment_driver.py')
    (out/'toolchain.json').write_text(json.dumps(tools,indent=2)+'\n')
    source=package['root']/(package['native']+'.sp');text=source.read_text()
    definitions,ports=hierarchy(text);cell_paths=[];mos_count=0
    if args.active_columns==1:
        top=package['native'].lower();stacked='stacked_colgrp_x256x32x1'
        children=[f for f in definitions[stacked]if f[0].lower().startswith('x')]
        if len(children)!=32 or any(f[-1].lower()!='colgrp_sram_6t122'for f in children):
            raise ValueError('unexpected native column hierarchy')
        definitions[stacked]=[children[0]]
        definitions[top]=[f for f in definitions[top]if f[0].lower()!='xdata_bottom']
        # Retain the real controller and one complete 1024-bit column with
        # its real sense/write/precharge circuits. Other columns' loads are
        # absent, so this reduced circuit cannot qualify macro timing.
        import textwrap
        emitted=[text[:text.lower().index('.subckt')]]
        for name,body in definitions.items():
            for fields in [['.SUBCKT',name,*ports[name]],*body,['.ENDS',name]]:
                chunks=textwrap.wrap(' '.join(fields),width=120,break_long_words=False,break_on_hyphens=False)
                emitted.append(chunks[0]);emitted.extend('+ '+part for part in chunks[1:])
        text='\n'.join(emitted)+'\n'
    def visit(name,path):
        nonlocal mos_count
        if name=='sram_cell_6t_122':cell_paths.append(path)
        for fields in definitions[name]:
            if fields[0].lower().startswith('m'):mos_count+=1
            elif fields[0].lower().startswith('x'):visit(fields[-1].lower(),path+'.'+fields[0].lower())
    visit(package['native'].lower(),'xmacro')
    if len(cell_paths)!=1024*args.active_columns:raise ValueError('native array capacity differs from the selected experiment')
    lines=[]
    for line in text.splitlines():
        if line.startswith('M'):
            names={k.lower()for k in re.findall(r'(\w+)\s*=',line)}
            if names!={'l','w','nfin'}:raise ValueError('unsupported native MOS parameters')
            line='N'+line[1:];line=re.sub(r'(?i)\s+w\s*=\s*\S+','',line)
        lines.append(line)
    (out/'macro.inc').write_text('\n'.join(lines)+'\n')
    model=package['root']/'models'/('7nm_'+args.corner+'.pm')
    (out/'models.inc').write_text(model_port(model.read_text()))
    point=operating_point(profile(args.corner)['liberty']);voltage=point['values']['nom_voltage'];temp=point['values']['nom_temperature']
    mapping={'vdd':'vdd','vss':'0','clk':'clk','ce_n':'0','we_n':'we','oe_n':'0'}
    for i in range(4):mapping[f'sdel[{i}]']='0'
    for i in range(10):mapping[f'a[{i}]']='0'
    for i in range(64):
        mapping[f'd[{i}]']='data'
        mapping[f'q[{i}]']='q'+str(i)if i<args.active_columns else'0'
    native_ports=ports[package['native'].lower()]
    if set(p.lower()for p in native_ports)!=set(mapping):raise ValueError('native port map differs')
    ramp=args.startup=='supply_ramp'
    supply_source='PWL(0 0 100p {supply})'if ramp else'{supply}'
    data_source='PWL(0 0 100p {supply} 4n {supply} 4.01n 0)'if ramp else'PWL(0 {supply} 4n {supply} 4.01n 0)'
    script=f'''Full ASAP7 SRAM schematic, address-zero two-polarity diagnostic
.include models.inc
.include macro.inc
.temp {temp}
.param supply={voltage}
.options tnom=25
{'.options rshunt='+str(args.solver_rshunt_ohm) if args.solver_rshunt_ohm else ''}
VDD vdd 0 {supply_source}
VCLK clk 0 PULSE(0 {{supply}} .5n 10p 10p .99n 2n)
VWE we 0 PWL(0 0 2n 0 2.01n {{supply}} 4n {{supply}} 4.01n 0 6n 0 6.01n {{supply}})
VDATA data 0 {data_source}
Xmacro {' '.join(mapping[p.lower()]for p in native_ports)} {package['native']}
'''
    for i in range(args.active_columns):script+=f'CQ{i} q{i} 0 2f\n'
    for path in cell_paths:script+=f'.ic v({path}.q)=0 v({path}.qb)={"0"if ramp else"{supply}"}\n'
    vectors=' '.join(f'v(q{i})'for i in range(args.active_columns))
    script+=f'''.tran 5p 8n {'uic'if args.startup in('uic','supply_ramp')else''}
.control
set wr_singlescale
set wr_vecnames
set numdgt=15
save {vectors}
run
wrdata voltages.txt {vectors}
quit
.endc
.end
'''
    (out/'macro.cir').write_text(script);init=out/'init';init.mkdir()
    (init/'spinit').write_text('* Local full-SRAM schematic experiment.\n')
    (out/'.spiceinit').write_text('option klu\nosdi '+tools['osdi']+'\n')
    log=out/'ngspice.log';timeout=False
    with log.open('w')as stream:
        process=subprocess.Popen(['stdbuf','-oL','-eL',tools['ngspice'],'-b','macro.cir'],cwd=out,
            stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'SPICE_SCRIPTS':str(init)})
        (out/'running.json').write_text(json.dumps(dict(pid=process.pid,timeout_seconds=args.timeout,
            macro_package_manifest_sha256=package['manifest_sha256'],array_bitcells=len(cell_paths),transistors=mos_count))+'\n')
        try:code=process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timeout=True;process.kill();code=process.wait()
    samples=[];wave=out/'voltages.txt'
    if wave.exists():samples=[list(map(float,line.split()))for line in wave.read_text().splitlines()[1:]]
    checks=[]
    if samples:
        for time,one in((3.9,True),(7.9,False)):
            row=min(samples,key=lambda r:abs(r[0]-time*1e-9))
            good=len(row)==args.active_columns+1 and abs(row[0]-time*1e-9)<=1e-11 and all(
                v>.8*voltage if one else v<.2*voltage for v in row[1:])
            checks.append(dict(time_ns=time,actual_sample_time_ns=row[0]*1e9,expected_word='ffffffffffffffff'if one else'0000000000000000',
                voltage_min_V=min(row[1:]),voltage_max_V=max(row[1:]),passed=good))
    clean=not re.search(r'(?im)^(?:warning:|error|.*aborted|.*out of bounds)',log.read_text())
    result=dict(kind='full_native_sram_schematic_transistor_experiment'if args.active_columns==64 else'reduced_native_column_transistor_experiment',
        passed=not timeout and code==0 and clean and len(checks)==2 and all(c['passed']for c in checks),
        timed_out=timeout,returncode=code,timeout_seconds=args.timeout,
        macro_package_manifest_sha256=package['manifest_sha256'],source_schematic_sha256=sha(source),
        corner=args.corner,voltage_V=voltage,temperature_C=temp,bitcells=len(cell_paths),transistors=mos_count,
        native_clock_period_ns=2,initialized_power_on_pattern=f'Q=0,QB={"0"if ramp else"VDD"},all{len(cell_paths)} retained cells',
        initialization_method=args.startup,
        resistance_formulation=args.resistance_formulation,
        supply_ramp_time_ps=100 if ramp else None,
        solver_rshunt_ohm=args.solver_rshunt_ohm,
        solver_shunt_is_physical=False,
        data_columns=args.active_columns,full_width_retained=args.active_columns==64,
        checked_addresses=[0],checks=checks,waveform_rows=len(samples),
        schematic_only=True,layout_parasitics_included=False,physical_frequency_certified=False,
        whole_macro_characterized=False,full_macro_drc_lvs_verified=False,
        evidence={str(p):sha(p)for p in out.rglob('*')if p.is_file()})
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='evidence'},indent=2))
    if not result['passed']:raise RuntimeError('full SRAM schematic check failed or reached its bounded runtime')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sram-package',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--corner',choices=['TT','SS','FF'],default='TT')
    p.add_argument('--startup',choices=['uic','operating_point','supply_ramp'],default='uic')
    p.add_argument('--active-columns',type=int,choices=[1,64],default=64,
        help='one real column isolates convergence/peripheral behavior; its reduced loading cannot certify the macro')
    p.add_argument('--solver-rshunt-ohm',type=float,
        help='explicit numerical shunt for convergence diagnosis only; not physical leakage or signoff evidence')
    p.add_argument('--resistance-formulation',choices=['current','voltage','zero_sheet'],default='current',
        help='separate equivalent voltage-branch OSDI model; original model and all resistance values are retained')
    p.add_argument('--timeout',type=int,default=600);args=p.parse_args()
    if not 1<=args.timeout<=3600:p.error('timeout must be within 1..3600 seconds')
    if args.solver_rshunt_ohm is not None and (not math.isfinite(args.solver_rshunt_ohm) or args.solver_rshunt_ohm<1e9):
        p.error('diagnostic shunt resistance must be finite and at least 1e9 ohm')
    run(args)
