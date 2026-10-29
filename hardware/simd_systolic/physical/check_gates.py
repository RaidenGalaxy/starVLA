#!/usr/bin/env python3
"""Check mapped arithmetic against the rational oracle using Liberty models."""
import argparse
import fcntl
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from simd_eval.reference import Format,Op,fp_operation
from simd_eval.architecture import require_f2
from simd_eval.cycle_model import mesh_trace_cycles,verify_mesh_cycles
sys.path.insert(0,str(ROOT/'scripts'))
from check_pe import make_vectors,verify_output
from check_pe_state import state_vectors
from check_pe_reset import verify_reset_coverage
from check_array import make_vectors as array_vectors,verify_output as verify_array
from check_array_events import make_vectors as event_vectors,verify as verify_events


def check_array(out,identity,routed=False,jobs=4,hierarchical=False,sim_threads=1,timeout=None):
    """Check the mapped mesh, wave alignment and pipelined observation ports."""
    if hierarchical and(routed or not identity.get('pe_hierarchy')):
        raise ValueError('hierarchical array gate simulation requires preserved PE hierarchy in the synthesis netlist')
    kind='routed'if routed else'synthesis'
    suffix=('_hier'if hierarchical else'')+(f'_mt{sim_threads}'if sim_threads!=1 else'')
    work=out/('gate_'+kind+suffix);work.mkdir(exist_ok=True)
    guard=(work/'.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    net=out/('routed.v'if routed else'synth.v');lib=Path(identity['liberty_file'])
    if digest(lib)!=identity['liberty_sha256']:raise ValueError('physical process library changed')
    evidence=json.loads((out/'timing.json').read_text())if routed else identity
    if digest(net)!=evidence['netlist_sha256'if routed else'mapped_sha256']:raise ValueError('mapped array netlist changed')
    cells=work/'cells.v';netcopy=work/'netlist.v';shutil.copy2(net,netcopy)
    if digest(netcopy)!=digest(net):raise ValueError('mapped array changed during snapshot')
    subprocess.run(['yosys','-Q','-T','-p',f'read_liberty -ignore_miss_func {lib}; write_verilog {cells}'],check=True,stdout=subprocess.DEVNULL)
    simulation_net=work/'simulation_netlist.v';script=work/'splitnets.ys'
    script.write_text(f'read_liberty -lib {lib}\nread_verilog {netcopy}\nhierarchy -check -top SystolicArray\nsplitnets\nwrite_verilog -noattr -noexpr {simulation_net}\n')
    with(work/'splitnets.log').open('w')as log:
        subprocess.run(['yosys','-Q','-T','-s',str(script)],check=True,stdout=log,stderr=subprocess.STDOUT)
    variant=digest(simulation_net)[:12];binary=work/('sim_'+variant)
    caps=json.loads((out/'generated/capabilities.json').read_text());rows,cols=caps['rows'],caps['cols']
    timeout=timeout if timeout is not None else 21600 if rows*cols>4 else 3600
    if timeout<=0:raise ValueError('simulation timeout must be positive')
    header=f'#define SIMULATION_THREADS {sim_threads}\n#define ROWS {rows}\n#define COLS {cols}\n#define READ_LATENCY {caps["array_read_latency"]}\n'
    for name,n,port in [('A',rows,'a'),('B',cols,'b'),('Mask',cols,'masks')]:
        header+=f'inline void set{name}(VSystolicArray& d,int i,uint64_t x) {{switch(i){{'
        header+=''.join(f'case {i}:d.io_{port}_{i}=x;break;'for i in range(n))+'}}\n'
    (work/'ports.hpp').write_text(header)
    for source in(ROOT/'sim/array_main.cpp',ROOT/'sim/wide.hpp',ROOT/'sim/assertions.hpp',ROOT/'sim/coverage.hpp'):shutil.copy2(source,work/source.name)
    hierarchy=[]
    if hierarchical:
        configuration=work/'models.vlt'
        configuration.write_text('`verilator_config\nhier_block -module "SimdPE"\nhier_workers -module "SimdPE" -workers 1\n')
        hierarchy=['--hierarchical',str(configuration)]
    parallel=['--threads',str(sim_threads),'--hierarchical-threads',str(sim_threads),'--stats']if sim_threads!=1 else[]
    with(work/'build.log').open('w')as log:
        subprocess.run(['verilator','--cc','--exe','--build','-j',str(jobs),'-Wno-fatal','--top-module','SystolicArray',
          '--Mdir',str(work/('obj_'+variant)),'-CFLAGS',f'-O1 -std=c++17 -I{work}',str(cells),str(simulation_net),
          str(work/'array_main.cpp'),'-o',str(binary),*hierarchy,*parallel],check=True,stdout=log,stderr=subprocess.STDOUT)
    commands,expected,macs=array_vectors(caps,k=17,pipelined_read=True)
    trace=work/'vectors.txt';trace.write_text(''.join(commands))
    result=subprocess.run([str(binary),str(trace)],capture_output=True,text=True,timeout=timeout)
    (work/'simulation.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(result.stdout[-2000:]+result.stderr)
    counts=verify_array(result.stdout,expected,macs)
    event_commands,event_states,event_counts=event_vectors(caps)
    event_trace=work/'event_vectors.txt';event_trace.write_text(''.join(event_commands))
    event_result=subprocess.run([str(binary),str(event_trace)],capture_output=True,text=True,timeout=timeout)
    (work/'event_simulation.log').write_text(event_result.stdout+event_result.stderr)
    if event_result.returncode:raise RuntimeError(event_result.stdout[-2000:]+event_result.stderr)
    event_check=verify_events(event_result.stdout,event_states,event_counts)
    # Include empty waves, a masked leading row and a drained format change.
    # Normal GEMM vectors always activate row zero and cannot expose a gap
    # between the intermediate registers of a two-clock neighbor hop.
    drain_commands=[];drain_expected={}
    for index,(fmt,row_mask)in enumerate([(4,0),(4,1<<(rows-1)),(10,1)]):
        value=1 if fmt==4 else 0x3ff0000000000000
        payload=[1]*cols+[value]*(rows+cols)
        row=rows-1 if index==1 else 0
        drain_commands.extend([f'I {fmt} {index} 1 {index} {row_mask:x} '+ ' '.join(f'{v:x}'for v in payload)+'\n',
                               'D\n',f'E {index}\n',f'R {index} {row} 0 {100+index}\n'])
        drain_expected[100+index]=value if row_mask else 0
    drain_commands.extend(['S 0 0 0 0 0\n','D\n','E 3\n'])
    drain_trace=work/'drain_vectors.txt';drain_trace.write_text(''.join(drain_commands))
    drain_result=subprocess.run([str(binary),str(drain_trace)],capture_output=True,text=True,timeout=timeout)
    (work/'drain_simulation.log').write_text(drain_result.stdout+drain_result.stderr)
    if drain_result.returncode:raise RuntimeError(drain_result.stdout[-2000:]+drain_result.stderr)
    verify_array(drain_result.stdout,drain_expected,2*cols)
    drain_check=verify_mesh_cycles(drain_result.stdout,mesh_trace_cycles(caps,drain_commands))
    report=dict(passed=True,scope='array',stages=identity['stages'],rows=rows,cols=cols,**counts,
      contexts=16,all_formats=True,pipelined_read=True,tail_masks=True,cross_tile_accumulation=True,
      mapped_netlist_kind=kind,rtl_sha256=identity['rtl_sha256'],netlist_sha256=digest(netcopy),
      liberty_sha256=digest(lib),simulator_sha256=digest(binary),trace_sha256=digest(trace),
      event_check=event_check,event_trace_sha256=digest(event_trace),
      wave_drain_protocol_checked=True,all_wave_stages_tracked=caps.get('all_wave_stages_tracked',False),
      wave_drain_check=drain_check,wave_drain_trace_sha256=digest(drain_trace),
      verilation_mode='hierarchical_pe'if hierarchical else'flat',
      simulation_root_threads=sim_threads,arithmetic_block_simulation_threads=1,
      host_timeout_seconds=timeout,
      simulation_netlist_sha256=digest(simulation_net),simulation_transform='Yosys splitnets (internal wire names only)',
      zero_delay_gate_simulation=True,timing_validation='separate extracted STA required')
    for name in('gate_verification.json',f'gate_verification_{kind}{suffix}.json'):
        (out/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


def check_pe(out,identity,routed=False,jobs=4):
    """Exercise the actual mapped PE, including state, masks and result stalls."""
    kind='routed'if routed else'synthesis';work=out/('gate_'+kind);work.mkdir(exist_ok=True)
    guard=(work/'.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    net=out/('routed.v'if routed else'synth.v');lib=Path(identity['liberty_file'])
    if digest(lib)!=identity['liberty_sha256']:raise ValueError('physical process library changed')
    evidence=json.loads((out/'timing.json').read_text())if routed else identity
    if digest(net)!=evidence['netlist_sha256'if routed else'mapped_sha256']:raise ValueError('mapped PE netlist changed')
    cells=work/'cells.v';netcopy=work/'netlist.v';shutil.copy2(net,netcopy)
    if digest(netcopy)!=digest(net):raise ValueError('mapped PE changed during snapshot')
    subprocess.run(['yosys','-Q','-T','-p',f'read_liberty -ignore_miss_func {lib}; write_verilog {cells}'],check=True,stdout=subprocess.DEVNULL)
    # Preserve the mapped cells and top ports, but give internal wire bits
    # separate names. Vector aliases in prefix trees otherwise appear as
    # combinational cycles to Verilator despite the acyclic bit-level graph.
    simulation_net=work/'simulation_netlist.v';script=work/'splitnets.ys'
    script.write_text(f'read_liberty -lib {lib}\nread_verilog {netcopy}\nhierarchy -check -top SimdPE\nsplitnets\nwrite_verilog -noattr -noexpr {simulation_net}\n')
    with(work/'splitnets.log').open('w')as log:
        subprocess.run(['yosys','-Q','-T','-s',str(script)],check=True,stdout=log,stderr=subprocess.STDOUT)
    variant=digest(simulation_net)[:12];binary=work/('sim_'+variant)
    caps=json.loads((out/'generated/capabilities.json').read_text())
    (work/'ports.hpp').write_text(f'#define READ_LATENCY {caps["pe_read_latency"]}\n#define SEED_LATENCY {caps.get("seed_write_latency",1)}\n'
        f'#define FIXED_STATE_COMMIT {int(caps.get("fixed_state_commit",False))}\n#define STATE_COMMIT_LATENCY {caps.get("state_commit_latency",0)}\n'
        f'#define REGISTERED_CONTEXT_ADMISSION {int(caps.get("registered_context_admission",False))}\n'
        f'#define PE_RESPONSE_CREDITS {caps.get("pe_response_credits",0)}\n')
    for source in(ROOT/'sim/pe_main.cpp',ROOT/'sim/wide.hpp',ROOT/'sim/coverage.hpp'):shutil.copy2(source,work/source.name)
    with(work/'build.log').open('w')as log:
        subprocess.run(['verilator','--cc','--exe','--build','-j',str(jobs),'-Wno-fatal','--top-module','SimdPE',
          '--Mdir',str(work/('obj_'+variant)),'-CFLAGS',f'-O1 -std=c++17 -I{work}',str(cells),str(simulation_net),
          str(work/'pe_main.cpp'),'-o',str(binary)],check=True,stdout=log,stderr=subprocess.STDOUT)
    commands,expected,acc,sticky=make_vectors();trace=work/'vectors.txt';trace.write_text(''.join(commands))
    result=subprocess.run([str(binary),str(trace)],capture_output=True,text=True,timeout=1200)
    (work/'simulation.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(result.stdout[-2000:]+result.stderr)
    minimum=identity['stages']+caps['pe_commit_latency']+caps['pe_state_read_latency']
    count=verify_output(result.stdout,expected,acc,sticky,minimum)
    state_trace=work/'state_vectors.txt';state_trace.write_text(state_vectors(commands,expected,caps))
    state=subprocess.run([str(binary),str(state_trace),'--state-stress'],capture_output=True,text=True,timeout=1200)
    (work/'state_simulation.log').write_text(state.stdout+state.stderr)
    if state.returncode:raise RuntimeError(state.stdout[-2000:]+state.stderr)
    verify_output(state.stdout,expected,acc,sticky,minimum)
    observed=[list(map(int,line.split()[1:]))for line in state.stdout.splitlines()if line.startswith('Q ')]
    if len(observed)!=1 or observed[0][1:3]!=[16,count]or observed[0][3]<=0:
        raise AssertionError('mapped PE live state coverage missing')
    registered=caps.get('registered_context_admission',False)
    admission=[list(map(int,line.split()[1:]))for line in state.stdout.splitlines()if line.startswith('A ')]
    if registered and(len(admission)!=1 or len(admission[0])!=2 or min(admission[0])<=0):
        raise AssertionError('mapped PE did not exercise queued context dependencies')
    reset_coverage={}
    if registered:
        reset=subprocess.run([str(binary),str(trace),'--reset-stress'],capture_output=True,text=True,timeout=1200)
        (work/'reset_simulation.log').write_text(reset.stdout+reset.stderr)
        if reset.returncode:raise RuntimeError(reset.stdout[-2000:]+reset.stderr)
        reset_coverage=verify_reset_coverage(reset.stdout,caps)
        verify_output(reset.stdout,expected,acc,sticky,minimum)
    report=dict(passed=True,scope='pe',stages=identity['stages'],vectors=count,contexts=16,
      all_formats=True,dependent_issues=True,long_result_stalls=True,sticky_flags=True,
      mapped_netlist_kind=kind,rtl_sha256=identity['rtl_sha256'],netlist_sha256=digest(netcopy),
      liberty_sha256=digest(lib),simulator_sha256=digest(binary),trace_sha256=digest(trace),
      simulation_netlist_sha256=digest(simulation_net),simulation_transform='Yosys splitnets (internal wire names only)',
      live_state_observations=observed[0][0],seeded_contexts=16,disabled_read_clocks=observed[0][3],
      fixed_state_commit=caps.get('fixed_state_commit',False),state_trace_sha256=digest(state_trace),
      registered_context_admission=registered,
      queued_dependency_admissions=admission[0][0]if registered else 0,
      maximum_admission_wait_clocks=admission[0][1]if registered else 0,
      reset_recovery=reset_coverage,
      state_oracle='independent rational arithmetic and input-derived admission calendar'if registered else
        'independent rational arithmetic at fixed commit'if caps.get('fixed_state_commit')else'accepted responses',
      zero_delay_gate_simulation=True,timing_validation='separate extracted STA required')
    for name in('gate_verification.json',f'gate_verification_{kind}.json'):
        (out/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


def check(out,stages,count,routed=False,jobs=4,hierarchical_array=False,sim_threads=1,timeout=None):
    require_f2(stages)
    top='SimdFpF264'if stages==2 else'SimdFpHF64'if stages==8 else'SimdFpHF12_64'
    identity=json.loads((out/'build_identity.json').read_text())
    if identity['scope']=='array':
        if identity['stages']!=stages:raise ValueError('pipeline depth differs from physical array')
        return check_array(out,identity,routed,jobs,hierarchical_array,sim_threads,timeout)
    if sim_threads!=1 or timeout is not None:raise ValueError('thread/timeout options currently apply to array gate simulation')
    if identity['scope']=='pe':
        if identity['stages']!=stages:raise ValueError('pipeline depth differs from physical PE')
        return check_pe(out,identity,routed,jobs)
    if identity['scope']!='fp64' or identity['stages']!=stages:raise ValueError('gate harness requires the matching FP64 probe')
    free=identity.get('fp64_free_running',False)
    if free:top=identity['top']
    cells=out/'cells.v'
    lib=Path(identity['liberty_file'])
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(lib)!=identity['liberty_sha256']:raise ValueError('physical process library changed')
    subprocess.run(['yosys','-Q','-T','-p',f'read_liberty -ignore_miss_func {lib}; write_verilog {cells}'],check=True,stdout=subprocess.DEVNULL)
    wrapper=out/'GateDut.sv'
    wrapper.write_text(f'module GateDut(input clock,ce,input [63:0]a,b,c,input [2:0]op,output [63:0]result,output [4:0]flags);{top} u(.*);endmodule\n')
    net=out/('routed.v'if routed else'synth.v')
    proof=json.loads((out/'timing.json').read_text())if routed else identity
    if digest(net)!=proof['netlist_sha256'if routed else'mapped_sha256']:raise ValueError('physical netlist identity changed')
    subprocess.run(['verilator','--cc','--exe','--build','-j',str(jobs),'-Wno-fatal','--top-module','GateDut',
      '--Mdir',str(out/'gate_obj'),str(cells),str(net),str(wrapper),str(ROOT/'tests/scalar_gate.cpp'),
      '-o',str(out/'gate_sim')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    rng=random.Random(981);vectors=[]
    edges=[0,1,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,
           0x7ff8000000000001,0x7ff0000000000001,0xfffffffffffff,0x10000000000000,0x7fefffffffffffff]
    for op in range(4):
      for i in range(count):
        a,b,c=[rng.getrandbits(64)for _ in range(3)]
        if i<len(edges)**2:a,b=edges[i%len(edges)],edges[i//len(edges)]
        expected,flags=fp_operation(a,b,c,Format.F64,Op(op))
        vectors.append(f'{a:x} {b:x} {c:x} {op} {expected:x} {flags}\n')
    vf=out/'gate_vectors.txt';vf.write_text(''.join(vectors))
    run=subprocess.run([str(out/'gate_sim'),str(vf),str(stages),'free'if free else'elastic'],check=True,text=True,capture_output=True)
    report=json.loads(run.stdout);report['netlist_sha256']=hashlib.sha256(net.read_bytes()).hexdigest()
    report.update(scope='fp64',stages=stages,rtl_sha256=identity['rtl_sha256'],mapped_netlist_kind='routed'if routed else'synthesis',random_clock_enable=not free)
    report.update(liberty_sha256=digest(lib),simulator_sha256=digest(out/'gate_sim'),
        trace_sha256=digest(vf),reference_sha256=digest(ROOT/'python/simd_eval/reference.py'),
        testbench_sha256=digest(ROOT/'tests/scalar_gate.cpp'),test_script_sha256=digest(__file__),
        zero_delay_gate_simulation=True,timing_validation='separate extracted STA required')
    (out/'gate_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--stages',type=int,choices=[2],default=2)
    p.add_argument('--count',type=int,default=300);p.add_argument('--routed',action='store_true');p.add_argument('-j',type=int,default=4)
    p.add_argument('--hierarchical-array',action='store_true',help='compile the actual preserved mapped PE once; synthesis netlist only')
    p.add_argument('--sim-threads',type=int,choices=[1,2,4,8],default=1,help='host root-model workers; no change to netlist or hardware clocks')
    p.add_argument('--timeout',type=int,help='host limit per array simulation in seconds; defaults to 3600 for 2x2 and 21600 for larger arrays')
    a=p.parse_args();check(a.directory.resolve(),a.stages,a.count,a.routed,a.j,a.hierarchical_array,a.sim_threads,a.timeout)
