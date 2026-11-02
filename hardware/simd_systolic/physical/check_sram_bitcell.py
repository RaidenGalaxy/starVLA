#!/usr/bin/env python3
"""Exercise the actual six-transistor schematic with pinned BSIM-CMG107.

Ideal peripheral drivers and explicit 10fF bitlines isolate the cell. This is
not an extracted array, a SRAM timing model, or a replacement for macro signoff.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'physical'),str(ROOT/'python')]
from asap7_platform import profile
from prepare_analog import prepare,sha
from pvt import operating_point
from sram_package import load_package
from simd_eval.architecture import ARCH


def model_port(text):
    text,count=re.subn(r'(?mi)^\.model[ \t]+(\w+)[ \t]+(nmos|pmos)[ \t]+level[ \t]*=[ \t]*72[ \t]*$',
        lambda m:'.model '+m[1]+' bsimcmg\n+devtype='+('1'if m[2].lower()=='nmos'else'0'),text)
    if count!=8:raise ValueError('unexpected ASAP7 model declarations')
    text,count=re.subn(r'(?i)\bversion\s*=\s*107\b','',text)
    if count!=8:raise ValueError('model version is not exactly BSIM-CMG107')
    return text


def cell_port(text):
    match=re.search(r'(?ims)^\.subckt sram_cell_6t_122\b.*?^\.ends[^\n]*',text)
    if not match:raise ValueError('actual macro has no selected 6T cell')
    original=match[0];lines=[];devices=[]
    for line in original.splitlines():
        if line.startswith('M'):
            parts=line.split();parameters=dict((k.lower(),v)for k,v in re.findall(r'(\w+)\s*=\s*([^\s]+)',line))
            if set(parameters)!={'l','w','nfin'}or parts[5]not in('nmos_sram','pmos_sram'):
                raise ValueError('unexpected bitcell model or geometry')
            devices.append(dict(instance=parts[0],model=parts[5],parameters=parameters))
            # BSIM-CMG107 has no W parameter. Preserve its explicit L/NFIN;
            # the removed W is retained as an annotation in the report.
            line='N'+line[1:];line=re.sub(r'(?i)\s+w\s*=\s*\S+','',line)
        lines.append(line)
    if len(devices)!=6:raise ValueError('cell is not six transistors')
    return original,'\n'.join(lines)+'\n',devices


def deck(voltage,temperature):
    return f'''Actual ASAP7 6T cell, ideal peripheral drivers, two-ns operation spacing
.include models.inc
.include cell.inc
.param supply={voltage}
.temp {temperature}
.options tnom=25
VDD vdd 0 {{supply}}
VWL wl 0 PWL(0 0 .3n 0 .31n {{supply}} .7n {{supply}} .71n 0
+ 2.3n 0 2.31n {{supply}} 2.7n {{supply}} 2.71n 0
+ 4.3n 0 4.31n {{supply}} 4.7n {{supply}} 4.71n 0
+ 6.3n 0 6.31n {{supply}} 6.7n {{supply}} 6.71n 0)
VWR wr 0 PWL(0 0 .2n 0 .21n {{supply}} .8n {{supply}} .81n 0
+ 4.2n 0 4.21n {{supply}} 4.8n {{supply}} 4.81n 0)
VPRE pre 0 PWL(0 {{supply}} .1n {{supply}} .11n 0
+ 2n 0 2.01n {{supply}} 2.1n {{supply}} 2.11n 0
+ 6n 0 6.01n {{supply}} 6.1n {{supply}} 6.11n 0)
VWD wd 0 PWL(0 {{supply}} 4n {{supply}} 4.1n 0)
VWDN wdn 0 PWL(0 0 4n 0 4.1n {{supply}})
.model ideal_driver SW(RON=10 ROFF=1e15 VT={voltage/2} VH=0)
SWBL bl wd wr 0 ideal_driver
SWBLN bln wdn wr 0 ideal_driver
SPBL bl vdd pre 0 ideal_driver
SPBLN bln vdd pre 0 ideal_driver
CBL bl 0 10f
CBLN bln 0 10f
Xcell wl bln bl vdd 0 sram_cell_6t_122
.ic v(xcell.q)=0 v(xcell.qb)={{supply}} v(bl)={{supply}} v(bln)={{supply}}
.tran 1p 8n uic
.control
set wr_singlescale
set wr_vecnames
set numdgt=15
run
wrdata voltages.txt v(xcell.q) v(xcell.qb) v(bl) v(bln)
quit
.endc
.end
'''


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
    if out.exists():raise ValueError('preserve prior transistor diagnostics; choose a new --out')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'check_driver.py')
    shutil.copy2(ROOT/'physical/analog_tools.lock.json',out/'analog_tools.lock.json')
    (out/'toolchain.json').write_text(json.dumps(tools,indent=2)+'\n')
    original,ported,devices=cell_port((package['root']/(package['native']+'.sp')).read_text())
    (out/'original_cell.sp').write_text(original+'\n')
    records=[]
    for corner in('TT','SS','FF'):
        directory=out/corner;directory.mkdir();init=directory/'init';init.mkdir()
        (init/'spinit').write_text('* Isolated transistor diagnostic, with no global code models.\n')
        (directory/'.spiceinit').write_text(('option klu\n'if args.use_klu else'')+'osdi '+tools['osdi']+'\n')
        point=operating_point(profile(corner)['liberty'])
        if point['problems']:raise ValueError('missing standard-cell operating point')
        voltage=point['values']['nom_voltage'];temperature=point['values']['nom_temperature']
        model=package['root']/'models'/('7nm_'+corner+'.pm')
        (directory/'models.inc').write_text(model_port(model.read_text()))
        (directory/'cell.inc').write_text(ported)
        (directory/'cell.cir').write_text(deck(voltage,temperature))
        result=subprocess.run([tools['ngspice'],'-b','cell.cir'],cwd=directory,
            env={**os.environ,'SPICE_SCRIPTS':str(init)},capture_output=True,text=True,timeout=120)
        log=directory/'ngspice.log';log.write_text(result.stdout+result.stderr)
        wave=directory/'voltages.txt';samples=[]
        if wave.exists():
            for line in wave.read_text().splitlines()[1:]:
                row=list(map(float,line.split()))
                if len(row)!=5 or not all(map(math.isfinite,row)):raise ValueError('invalid analog waveform')
                samples.append(row)
        checks=[]
        def check(name,time_ns,one,read=False):
            row=min(samples,key=lambda row:abs(row[0]-time_ns*1e-9))
            if abs(row[0]-time_ns*1e-9)>2e-12:
                checks.append(dict(name=name,time_ns=time_ns,actual_sample_time_ns=row[0]*1e9,
                    passed=False,error='missing requested analog sample'))
                return
            _,q,qb,bl,bln=row
            stable=(q-qb if one else qb-q)>.2*voltage
            if not read:stable=stable and(q>.8*voltage and qb<.2*voltage if one else qb>.8*voltage and q<.2*voltage)
            if read:stable=stable and(bl-bln if one else bln-bl)>.1*voltage
            checks.append(dict(name=name,time_ns=time_ns,Q_V=q,QB_V=qb,BL_V=bl,BLN_V=bln,passed=stable))
        if samples:
            for name,t,one,read in [('write_one',.65,True,False),('hold_one',1.5,True,False),
                    ('read_one',2.6,True,True),('retain_after_read_one',3.2,True,False),
                    ('write_zero',4.65,False,False),('hold_zero',5.5,False,False),
                    ('read_zero',6.6,False,True),('retain_after_read_zero',7.2,False,False)]:
                check(name,t,one,read)
        clean=not re.search(r'(?im)^(?:warning:|error|.*aborted|.*out of bounds)',log.read_text())
        records.append(dict(corner=corner,voltage_V=voltage,temperature_C=temperature,
            source_model_sha256=sha(model),returncode=result.returncode,waveform_rows=len(samples),
            checks=checks,passed=result.returncode==0 and clean and len(checks)==8 and all(c['passed']for c in checks)))
    result=dict(kind='actual_6t_sram_cell_schematic_transistor_diagnostic',
        passed=all(r['passed']for r in records),macro_package_manifest_sha256=package['manifest_sha256'],
        native_macro=package['native'],bitcell='sram_cell_6t_122',devices=devices,
        simulator='ngspice42 with OpenVAF BSIM-CMG107.0.0',operating_points=records,
        resistance_formulation=args.resistance_formulation,
        linear_solver='KLU'if args.use_klu else'ngspice default',
        stimulus_operation_spacing_ns=2,external_bitline_capacitance_fF=10,
        ideal_peripheral_drivers=True,schematic_only=True,extracted_array=False,
        W_annotations_retained_in_report=True,physical_frequency_certified=False,
        full_macro_drc_lvs_verified=False,whole_macro_characterized=False,
        evidence={str(p):sha(p)for p in out.rglob('*')if p.is_file()})
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in('evidence','devices')},indent=2))
    if not result['passed']:raise RuntimeError('SRAM cell transistor checks failed; inspect preserved waveforms')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sram-package',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resistance-formulation',choices=['current','voltage','zero_sheet'],default='current')
    p.add_argument('--use-klu',action='store_true')
    run(p.parse_args())
