#!/usr/bin/env python3
"""Verify a specialized, authenticated I64 ALU component against the oracle."""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
from build import rtl_digest
from simd_eval.reference import Format,Op,operate
from run import sha


def run(directory,count,mapped,routed=False):
    directory=directory.resolve();identity=json.loads((directory/'build_identity.json').read_text())
    if identity['scope']!='int64'or identity['stages']!=2:raise ValueError('requires the specialized F2 I64 component')
    paths=[directory/'inputs'/name for name in identity['input_manifest']]
    for path in paths:
        if sha(path)!=identity['input_manifest'][path.name]:raise ValueError('component input changed: '+path.name)
    if rtl_digest(paths,directory/'inputs/common_cells')!=identity['rtl_sha256']:raise ValueError('component RTL identity changed')
    out=directory/('check_routed'if routed else'check_mapped'if mapped else'check_rtl');out.mkdir(exist_ok=False)
    shutil.copy2(__file__,out/'check_driver.py')
    bench=out/'int64_main.cpp';source=ROOT/'sim/alu_main.cpp';text=source.read_text()
    if text.count('d.io_in_bits_fmt=q.fmt;')!=1:raise ValueError('unexpected ALU harness')
    text=text.replace('VSimdAlu','VSimdInt64').replace('d.io_in_bits_fmt=q.fmt;','')
    bench.write_text(text);shutil.copy2(ROOT/'sim/coverage.hpp',out/'coverage.hpp')
    simulation_transform=None
    simulation_formatter=None
    if mapped or routed:
        net=directory/('routed.v'if routed else'synth.v');lib=Path(identity['liberty_file'])
        if sha(directory/'synth.v')!=identity['mapped_sha256']or sha(lib)!=identity['liberty_sha256']:
            raise ValueError('mapped component/library changed')
        actual_netlist_sha256=sha(net)
        if routed:
            timing=json.loads((directory/'timing.json').read_text())
            if timing['netlist_sha256']!=actual_netlist_sha256 or timing['rtl_sha256']!=identity['rtl_sha256']:
                raise ValueError('routed numerical check differs from extracted timing implementation')
            # Escaped scalar aliases such as foo[0] can collide with the
            # corresponding vector bit in Verilator after OpenROAD export.
            # Use the same wire-only split used by the PE/array gate checks.
            simulation_net=out/'tool_input.v';script=out/'splitnets.ys'
            script.write_text(f'read_liberty -lib {lib}\nread_verilog {net}\nhierarchy -check -top SimdInt64\nsplitnets\nwrite_verilog -noattr -noexpr {simulation_net}\n')
            with(out/'splitnets.log').open('w')as log:
                subprocess.run(['yosys','-Q','-T','-s',str(script)],check=True,stdout=log,stderr=subprocess.STDOUT)
            net=simulation_net
            simulation_transform='Yosys splitnets; wire representation only'
        elif identity.get('tool_input_netlist_sha256'):
            net=directory/'tool_input.v'
            if sha(net)!=identity['tool_input_netlist_sha256']:raise ValueError('imported tool netlist changed')
        else:
            # Synthesis-only builds have not reached OpenSTA's import step.
            # Yosys can emit bus assignments beyond Verilator's lexer limit;
            # keep the original mapping and wrap a private simulation copy.
            from format_mapped_verilog import wrap_assignments
            simulation_net=out/'tool_input.v';shutil.copy2(net,simulation_net)
            simulation_formatter=out/'netlist_formatter.py'
            shutil.copy2(ROOT/'physical/format_mapped_verilog.py',simulation_formatter)
            wrap_assignments(simulation_net,all_statements=True)
            net=simulation_net
            simulation_transform='whitespace only; identical non-whitespace token sequence'
        cells=out/'cells.v'
        subprocess.run(['yosys','-Q','-T','-p',f'read_liberty -ignore_miss_func {lib}; write_verilog {cells}'],check=True,stdout=subprocess.DEVNULL)
        inputs=[cells,net]
    else:inputs=paths
    with(out/'build.log').open('w')as log:
        subprocess.run(['verilator','--cc','--exe','--build','-j','4','-Wno-fatal','--top-module','SimdInt64',
            '--Mdir',str(out/'obj'),'-I'+str(directory/'inputs'),'-CFLAGS','-O1 -std=c++17',
            *map(str,inputs),str(bench),'-o',str(out/'sim')],check=True,stdout=log,stderr=subprocess.STDOUT)
    rng=random.Random(1264);vectors=[];expected=[]
    edges=[0,1,(1<<64)-1,1<<63,(1<<63)-1]
    accumulators=[0,1,(1<<144)-1,1<<143,(1<<143)-1]
    def add(op,a,b,acc,mask):
        tag=len(vectors);vectors.append(f'4 {op} {a:x} {b:x} {acc:x} {mask:x} {tag%16} {tag}\n')
        expected.append(operate(a,b,acc,Format.I64,Op(op),mask))
    for op in range(8):
        for a in edges:
            for b in edges:
                for acc in accumulators:
                    for mask in(0,1):add(op,a,b,acc,mask)
        for _ in range(count):add(op,rng.getrandbits(64),rng.getrandbits(64),rng.getrandbits(144),rng.randrange(2))
    trace=out/'vectors.txt';trace.write_text(''.join(vectors));checks=[]
    for mode in('continuous','stalled_idle'):
        env={k:v for k,v in os.environ.items()if k!='SIMD_ALU_IDLE_GAPS'}
        if mode=='stalled_idle':env['SIMD_ALU_IDLE_GAPS']='1'
        command=[str(out/'sim'),str(trace)]+(['101']if mode=='stalled_idle'else[])
        result=subprocess.run(command,capture_output=True,text=True,check=True,env=env)
        log=out/(mode+'.log');log.write_text(result.stdout+result.stderr);seen=set()
        for line in result.stdout.splitlines():
            if not line or not line[0].isdigit():continue
            tag,data,flags,overflow,illegal,start,end=line.split();tag=int(tag)
            if tag in seen or tag>=len(expected):raise AssertionError('duplicate or invalid response tag')
            if(int(data,16),int(flags,16),int(overflow,16))!=expected[tag]or int(illegal):
                raise AssertionError('I64 numerical mismatch at '+vectors[tag].strip())
            if mode=='continuous'and int(end)-int(start)!=2:raise AssertionError('I64 arithmetic latency is not two')
            seen.add(tag)
        if len(seen)!=len(vectors):raise AssertionError('missing I64 response')
        checks.append(dict(mode=mode,vectors=len(vectors),log_sha256=sha(log)))
    report=dict(passed=True,scope='specialized I64 component including conversion',pipeline_stages=2,
        arithmetic_cycles_without_stalls=2,random_seed=1264,checks=checks,rtl_sha256=identity['rtl_sha256'],
        parent_alu_rtl_sha256=identity['parent_alu_rtl_sha256'],mapped_netlist=mapped,
        routed_netlist=routed,zero_delay_numerical_check=True,
        netlist_sha256=actual_netlist_sha256 if mapped or routed else None,
        simulated_netlist_sha256=sha(net)if mapped or routed else None,
        simulation_transform=simulation_transform,
        simulation_formatter_sha256=sha(simulation_formatter)if simulation_formatter else None,
        simulator_sha256=sha(out/'sim'),trace_sha256=sha(trace),
        reference_sha256=sha(ROOT/'python/simd_eval/reference.py'),driver_sha256=sha(__file__),
        parent_harness_sha256=sha(source),physical_frequency_certified=False)
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--count',type=int,default=1000)
    group=p.add_mutually_exclusive_group();group.add_argument('--mapped',action='store_true');group.add_argument('--routed',action='store_true')
    a=p.parse_args();run(a.directory,a.count,a.mapped,a.routed)
