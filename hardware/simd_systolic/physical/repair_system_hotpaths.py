#!/usr/bin/env python3
"""Bounded signal-buffer ECOs selected from authenticated system STA paths.

Only the existing guarded per-net repair engine mutates the database. A round
finishing, or no longer finding extreme cell delays, never certifies timing.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from run import ROOT,sha
from netlist_fanout import PORT


def candidates(report,netlist,threshold,limit):
    slow={}
    pattern=re.compile(r'^\s+([-+\d.]+)\s+([-+\d.]+)\s+[\^v]\s+(\S+)/(Y|QN|Q|Z|ZN)\s+\(',re.M)
    for match in pattern.finditer(Path(report).read_text()):
        delay=float(match[1]);key=(match[3],match[4])
        if delay>threshold:slow[key]=max(delay,slow.get(key,0))
    header=re.compile(r'^\s*\w+\s+(\\\S+|[\w$]+)\s*\(')
    ports={};clock_nets=set();current=None
    with Path(netlist).open()as stream:
        for line in stream:
            match=header.match(line)
            if match:current=match[1].lstrip('\\')
            for pin in PORT.finditer(line):
                port,name,select=pin.groups();name=name.lstrip('\\')
                if select:name+=re.sub(r'\s','',select)
                if port in('CLK','clk','CK','clock'):clock_nets.add(name)
                if(current,port)in slow:ports[(current,port)]=name
            if re.search(r'\)\s*;',line):current=None
    selected=[];seen=set()
    for key,delay in sorted(slow.items(),key=lambda item:item[1],reverse=True):
        net=ports.get(key)
        if not net or net in seen or net in clock_nets:continue
        selected.append(dict(instance=key[0],port=key[1],net=net,cell_delay_ns=delay));seen.add(net)
        if len(selected)>=limit:break
    return selected


def run(args):
    out=args.out.resolve();source=args.source.resolve()
    if out.exists():raise ValueError('preserve earlier repair sequences')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'sequence_driver.py')
    shutil.copy2(ROOT/'physical/netlist_fanout.py',out/'fanout_parser.py')
    state=dict(kind='bounded_actual_system_hotpath_repairs',passed=False,rounds=[],
        core_period_ns=1,memory_period_ns=2,physical_frequency_certified=False,
        requested_rounds=args.rounds,maximum_nets_per_round=args.nets,
        threshold_cell_delay_ns=args.threshold,source_wait_seconds=args.wait_seconds)
    def save():
        p=out/'state.json.pending';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(out/'state.json')
    try:
        deadline=time.monotonic()+args.wait_seconds
        state.update(active_phase='waiting_for_authenticated_source',source=str(source));save()
        while True:
            try:
                json.loads((source/'verification.json').read_text());break
            except(FileNotFoundError,json.JSONDecodeError):pass
            if time.monotonic()>=deadline:raise RuntimeError('source completion record did not arrive')
            time.sleep(5)
        for index in range(1,args.rounds+1):
            proof=source/'verification.json';prior=json.loads(proof.read_text())
            if not prior['passed']or(prior['core_period_ns'],prior['memory_period_ns'])!=(1,2):
                raise ValueError('source has no completed diagnostic at target periods')
            if not prior['logical_roots_and_protected_data_clock_masters_preserved']:
                raise ValueError('source did not preserve guarded logic')
            report=source/'setup.rpt';netlist=source/'repaired.v'
            for p in(report,netlist):
                if prior['evidence'].get(str(p))!=sha(p):raise ValueError('source path/netlist evidence changed')
            targets=candidates(report,netlist,args.threshold,args.nets)
            selection=out/f'selection{index}.json'
            selection.write_text(json.dumps(dict(source_verification=str(proof),source_verification_sha256=sha(proof),
                source_setup_sha256=sha(report),source_netlist_sha256=sha(netlist),
                threshold_cell_delay_ns=args.threshold,targets=targets),indent=2)+'\n')
            if not targets:
                state['stopped_reason']='no_extreme_cell_outputs_in_reported_paths';break
            destination=out/f'round{index}';log=out/f'round{index}.log'
            command=[sys.executable,str(ROOT/'physical/repair_system_resets.py'),'--source',str(source),
                '--sram-package',str(args.sram_package.resolve()),'--out',str(destination),
                '--controls','--minimum','2','--limit',str(len(targets)),'--timeout','3600']
            for item in targets:command.extend(['--control-net',item['net']])
            state.update(active_phase=f'guarded_repair{index}',selection=str(selection));save()
            # The child engine owns and bounds its private OpenROAD group.
            with log.open('w')as stream:
                p=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                state['active_pid']=p.pid;save();code=p.wait()
            state.pop('active_pid',None)
            record=dict(selection=str(selection),selection_sha256=sha(selection),command=command,
                returncode=code,log=str(log),log_sha256=sha(log))
            state['rounds'].append(record);save()
            if code:raise RuntimeError('guarded hotpath round failed; evidence retained')
            result=json.loads((destination/'verification.json').read_text())
            record.update(verification_sha256=sha(destination/'verification.json'),timing=result['timing'])
            source=destination;save()
        state.update(passed=True,last_source=str(source),physical_frequency_certified=False)
    except Exception as error:
        state['error']=str(error);raise
    finally:
        state.pop('active_phase',None);save()
        (out/'verification.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--sram-package',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--nets',type=int,default=6)
    p.add_argument('--threshold',type=float,default=2);p.add_argument('--wait-seconds',type=int,default=3600)
    a=p.parse_args()
    if not 1<=a.rounds<=4 or not 1<=a.nets<=16 or a.threshold<1 or not 0<=a.wait_seconds<=3600:
        p.error('rounds1..4,nets1..16,threshold>=1ns,wait0..3600 required')
    run(a)
