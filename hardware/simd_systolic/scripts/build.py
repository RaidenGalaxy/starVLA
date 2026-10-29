#!/usr/bin/env python3
"""Build the actual all-format RTL once per profile; no precision specialization."""
import argparse,fcntl,json,os,subprocess,hashlib,sys,shutil,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from simd_eval.hardware_contract import hardware_contract
from simd_eval.architecture import ARCH, require_f2, validate_dram

def elaborate(kind,stages,rows,cols,dest,sram_latency=1):
    require_f2(stages)
    # Zinc's class directory is shared across profiles. Serialize elaboration,
    # then release the lock while independent C++/physical builds proceed.
    (ROOT/'.build').mkdir(exist_ok=True)
    with (ROOT/'.build/.elaboration.lock').open('a')as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        dest.mkdir(parents=True,exist_ok=True)
        for p in dest.glob('*.sv'):p.unlink()
        subprocess.run(['sbt','-J-Xmx8G','-batch',f'runMain simd.Generate {kind} {stages} {rows} {cols} {dest} {sram_latency}'],cwd=ROOT,check=True)

def sources():
    return [ROOT/'vendor/fpnew_pkg.sv',ROOT/'rtl/lzc.sv',
        *(ROOT/'vendor'/n for n in ['fpnew_classifier.sv','fpnew_rounding.sv','fpnew_fma.sv','fpnew_cast_multi.sv']),
        ROOT/'rtl/SimdFpF2.sv',ROOT/'rtl/SimdFpF2Core.sv',ROOT/'rtl/Asap7SramMacro.sv',ROOT/'rtl/SramBank.sv',ROOT/'rtl/AsyncFifo.sv',ROOT/'rtl/PrefixAdd.sv',ROOT/'rtl/PopulationCount.sv',ROOT/'rtl/ControlReplicas.sv',ROOT/'rtl/OperandReplicas.sv',ROOT/'rtl/SimdNarrow32.sv',ROOT/'rtl/OutputConvert.sv']

def rtl_digest(paths,headers=None):
    h=hashlib.sha256()
    for p in sorted(paths,key=lambda p:p.name):
        h.update(p.name.encode()+b'\0'+p.read_bytes())
    for p in sorted((headers or ROOT/'vendor/common_cells').glob('*.svh')):
        h.update(p.name.encode()+b'\0'+p.read_bytes())
    return h.hexdigest()

def build(kind='alu',stages=2,rows=1,cols=1,jobs=8,hierarchical=False,physical_snapshot=None,sram_latency=1,label=None,sim_threads=1,coverage=False,cohort=None):
    require_f2(stages)
    validate_dram(ROOT/'configs/ddr4.ini')
    frozen=None
    if cohort:
        from freeze_verification import verify
        frozen=verify(Path(cohort))
    if sim_threads not in(1,2,4,8):raise ValueError('simulation threads must be 1, 2, 4 or 8')
    if sim_threads!=1 and(not hierarchical or kind not in('array','system','gemm')):
        raise ValueError('parallel hierarchical simulation requires an array/system/GEMM target')
    if sram_latency!=ARCH['sram_read_latency']:raise ValueError('SRAM macro read latency must be one memory clock')
    if sram_latency!=1 and(kind not in('system','gemm')or physical_snapshot):raise ValueError('SRAM latency is a system/GEMM build parameter')
    if label is not None and not re.fullmatch(r'[a-z][a-z0-9_]{0,47}',label):raise ValueError('build label must be a lowercase identifier')
    identity=None
    if physical_snapshot:
        physical_snapshot=Path(physical_snapshot).resolve()
        identity=json.loads((physical_snapshot/'build_identity.json').read_text())
        if identity.get('architecture_sha256')!=ARCH['architecture_sha256']:
            raise ValueError('physical snapshot belongs to another architecture')
        if (identity['scope'],identity['stages'],identity['rows'],identity['cols'])!=(kind,stages,rows,cols):
            raise ValueError('physical snapshot scope/stages/dimensions differ from requested build')
    suffix='_'+identity['rtl_sha256'][:12]if identity else''
    if sram_latency!=1:suffix+=f'_sram{sram_latency}'
    if label:suffix+='_'+label
    if coverage:suffix+='_coverage'
    if sim_threads!=1:suffix+=f'_mt{sim_threads}'
    dest=ROOT/f'.build/{"qualified_"if identity else""}{kind}_{stages}_{rows}x{cols}{suffix}'
    dest.mkdir(parents=True,exist_ok=True)
    guard=(dest/'.lock').open('w');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    incomplete=dest/'.incomplete';incomplete.write_text('build has not completed\n')
    if not identity:elaborate(kind,stages,rows,cols,dest,sram_latency)
    top={'gemm':'GemmSystem','write_dma':'WriteDma','dma':'ReadDma','alu':'SimdAlu','pe':'SimdPE','array':'SystolicArray','system':'TileSystem'}[kind]
    snapshot=dest/'inputs';snapshot.mkdir(exist_ok=True)
    inputs=[]
    if identity:
        original=physical_snapshot/'inputs';paths=sorted(original.glob('*.sv'))
        digest=rtl_digest(paths,original/'common_cells')
        if digest!=identity['rtl_sha256']:raise RuntimeError('physical RTL snapshot changed')
        shutil.copy2(physical_snapshot/'generated/capabilities.json',dest/'capabilities.json')
        headers=original/'common_cells'
    else:
        paths=sources()+sorted(dest.glob('*.sv'));digest=rtl_digest(paths);headers=ROOT/'vendor/common_cells'
    for p in paths:
        target=snapshot/p.name;target.write_bytes(p.read_bytes());inputs.append(target)
    shutil.copytree(headers,snapshot/'common_cells',dirs_exist_ok=True)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in inputs}
    (dest/'rtl_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    cap=json.loads((dest/'capabilities.json').read_text());cap['rtl_sha256']=digest
    cap.update(ARCH)
    cap['sram_macro_used']=kind in ('system','gemm')
    cap['storage_implementation']='asap7_sram_macro' if cap['sram_macro_used'] else 'registers'
    cap['sram_physical_certified']=False
    cap['dram_config_sha256']=hashlib.sha256((ROOT/'configs/ddr4.ini').read_bytes()).hexdigest()
    cap['dependency_lock_sha256']=hashlib.sha256((ROOT/'dependencies.lock.json').read_bytes()).hexdigest()
    for key in ('rows','cols'):
        cap[key]={'rows':rows,'cols':cols}[key]
    cap['hardware_contract']=hardware_contract(cap)
    for emitted,expected in zip(cap.get('format_capabilities',[]),cap['hardware_contract']['formats']):
        if any(emitted[k]!=expected[k]for k in emitted):raise ValueError('generated RTL arithmetic capability differs from software contract')
    cap['code_coverage_instrumented']=coverage
    cap['code_coverage_types']=['line','expression','toggle']if coverage else[]
    cap['code_coverage_max_width']=512 if coverage else None
    if frozen:cap['verification_cohort_sha256']=frozen['cohort_sha256']
    cap['chisel_assertion_sites']=sum(p.read_text().count('$error("Assertion failed')for p in inputs)
    if cap.get('inline_chisel_assertions')and not cap['chisel_assertion_sites']:
        raise RuntimeError('Chisel assertions were advertised but absent from the emitted RTL')
    cap['simulation_threads']=sim_threads
    cap['simulation_root_threads']=sim_threads
    cap['arithmetic_block_simulation_threads']=1
    if label:cap['build_label']=label
    if kind=='system':cap['dma_overlap_observation']=True
    if kind=='pe':cap['live_state_observation_check']=True
    if kind in('array','system','gemm'):cap['checked_verilator_stop']=True
    if identity:cap['qualified_physical_snapshot']=str(physical_snapshot)
    main=ROOT/'sim'/f'{kind}_main.cpp'
    if kind=='array':
        header=f'#define ROWS {rows}\n#define COLS {cols}\n#define READ_LATENCY {cap["array_read_latency"]}\n'
        for name,n,port in [('A',rows,'a'),('B',cols,'b'),('Mask',cols,'masks')]:
            header+=f'inline void set{name}(VSystolicArray& d,int i,uint64_t x) {{switch(i){{'
            header+=''.join(f'case {i}:d.io_{port}_{i}=x;break;' for i in range(n))+'}}\n'
        (dest/'ports.hpp').write_text(header)
    if kind=='pe':(dest/'ports.hpp').write_text(f'#define READ_LATENCY {cap["pe_read_latency"]}\n#define SEED_LATENCY {cap.get("seed_write_latency",1)}\n'
        f'#define FIXED_STATE_COMMIT {int(cap.get("fixed_state_commit",False))}\n#define STATE_COMMIT_LATENCY {cap.get("state_commit_latency",0)}\n'
        f'#define REGISTERED_CONTEXT_ADMISSION {int(cap.get("registered_context_admission",False))}\n'
        f'#define PE_RESPONSE_CREDITS {cap.get("pe_response_credits",0)}\n')
    if kind=='gemm':(dest/'ports.hpp').write_text(f'#define HAS_PHASE_COUNTERS {int(bool(cap.get("sequencer_cycle_categories")))}\n#define HAS_SCRATCHPAD {int(bool(cap.get("scratchpad_access")))}\n#define HAS_NATIVE_PREFETCH {int(bool(cap.get("autonomous_prefetch")))}\n#define HAS_OUTPUT_EPILOGUE {int(bool(cap.get("output_epilogue")))}\n')
    if kind=='system':
        header=f'#define ROWS {rows}\n#define COLS {cols}\n#define READ_LATENCY {cap["array_read_latency"]}\n'
        header+=f'#define HAS_AGGREGATE_INSPECTION {int(bool(cap.get("aggregate_inspection")))}\n'
        header+='#define HAS_SYNC_SRAM 1\n'
        header+=f'#define HAS_OUTPUT_PUSH_OVERRIDE {int(bool(cap.get("output_push_override")))}\n'
        header+='inline void setCommandMask(VTileSystem& d,int i,uint64_t x) {switch(i){'
        header+=''.join(f'case {i}:d.io_command_bits_masks_{i}=x;break;' for i in range(cols))+'}}\n'
        (dest/'ports.hpp').write_text(header)
    if kind in('array','system','gemm'):
        header=dest/'ports.hpp';header.write_text(f'#define SIMULATION_THREADS {sim_threads}\n'+header.read_text())
    if main.exists():
        bench=dest/'testbench';bench.mkdir(exist_ok=True)
        bench_inputs=[main,*sorted((ROOT/'sim').glob('*.hpp'))]
        if kind in('system','gemm'):
            dep=ROOT/'.deps/dramsim3'
            if not(dep/'libdramsim3.so').exists()or not(dep/'src/dramsim3.h').exists():
                subprocess.run([sys.executable,str(ROOT/'scripts/prepare_dramsim.py')],check=True)
            # The public interface header is self-contained. Freeze it beside
            # the harness and authenticate it, without applying a DRAM include
            # flag to every unrelated arithmetic C++ translation unit.
            bench_inputs.append(dep/'src/dramsim3.h')
        if (dest/'ports.hpp').exists():bench_inputs.append(dest/'ports.hpp')
        control=None
        if hierarchical:
            block='SimdAlu'if kind=='pe'else'SimdPE'
            if kind not in ('pe','array','system','gemm'):raise ValueError('hierarchical mode requires a PE or array target')
            control=bench/'models.vlt'
            control.write_text('`verilator_config\nhier_block -module "'+block+'"\n')
            if sim_threads!=1:
                with control.open('a')as configuration:configuration.write('hier_workers -module "'+block+'" -workers 1\n')
            bench_inputs.append(control)
        for p in bench_inputs:
            if p.resolve()!=(bench/p.name).resolve():shutil.copy2(p,bench/p.name)
        cap['testbench_sha256']=rtl_digest(bench_inputs,headers=bench)
        cap['testbench_manifest']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in bench_inputs}
        cmd=['verilator','--cc','--exe','--build','-j',str(jobs),'--assert',
             '-Wno-fatal','-Wno-ASCRANGE','-Wno-WIDTHEXPAND','-Wno-WIDTHTRUNC',
             '--top-module',top,'--Mdir',str(dest/('obj_hier'if hierarchical else'obj')),'-I'+str(snapshot),
             # Quoted harness includes resolve beside the frozen .cpp file.
             # A build-label-specific -I on every generated model otherwise
             # invalidates ccache for unchanged arithmetic across iterations.
             '-CFLAGS','-O2 -std=c++17','-o',str(dest/'sim'),
             *map(str,inputs),str(bench/main.name)]
        cap['verilation_mode']='hierarchical'if hierarchical else'flat'
        if coverage:cmd+=['--coverage-line','--coverage-expr','--coverage-toggle','--coverage-underscore','--coverage-max-width','512']
        if hierarchical:cmd+=['--hierarchical',str(control)]
        if sim_threads!=1:cmd+=['--threads',str(sim_threads),'--hierarchical-threads',str(sim_threads),'--stats']
        if kind in ('system','gemm'):
            dep=ROOT/'.deps/dramsim3'
            if not (dep/'libdramsim3.so').exists():
                subprocess.run([sys.executable,str(ROOT/'scripts/prepare_dramsim.py')],check=True)
            libraries=dest/'libraries';libraries.mkdir(exist_ok=True)
            pending=libraries/'libdramsim3.so.tmp'
            shutil.copy2(dep/'libdramsim3.so',pending);pending.replace(libraries/'libdramsim3.so')
            cap['dram_library_sha256']=hashlib.sha256((libraries/'libdramsim3.so').read_bytes()).hexdigest()
            cap['dma_backend']='rtl'
            cmd+=['-LDFLAGS',f'-L{libraries} -Wl,-rpath,{libraries} -ldramsim3']
        subprocess.run(cmd,cwd=ROOT,check=True)
    else:
        subprocess.run(['verilator','--lint-only','--assert','-Wno-fatal','--top-module',top,
                        '-I'+str(snapshot),*map(str,inputs)],cwd=ROOT,check=True)
    if (dest/'sim').exists():cap['simulator_sha256']=hashlib.sha256((dest/'sim').read_bytes()).hexdigest()
    if frozen:verify(Path(cohort))
    tmp=dest/'capabilities.json.tmp';tmp.write_text(json.dumps(cap,indent=2)+'\n');tmp.replace(dest/'capabilities.json')
    incomplete.unlink()
    return dest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['gemm','alu','pe','array','system','dma','write_dma'],default='alu')
    p.add_argument('--stages',type=int,choices=[2],default=2);p.add_argument('--rows',type=int,default=1)
    p.add_argument('--cols',type=int);p.add_argument('-j',type=int,default=8);p.add_argument('--hierarchical',action='store_true')
    p.add_argument('--sim-threads',type=int,choices=[1,2,4,8],default=1,help='host simulation workers; does not change RTL clocks or arithmetic')
    p.add_argument('--physical-snapshot',type=Path);p.add_argument('--sram-latency',type=int,default=1)
    p.add_argument('--coverage',action='store_true',help='independent line/expression/toggle coverage build; SIMD_CODE_COVERAGE selects each output file')
    p.add_argument('--cohort',type=Path,help='refuse source changes before or during this build')
    p.add_argument('--label',help='retain an independent simulator snapshot for long evaluations');a=p.parse_args()
    print(build(a.kind,a.stages,a.rows,a.cols or a.rows,a.j,a.hierarchical,a.physical_snapshot,a.sram_latency,a.label,a.sim_threads,a.coverage,a.cohort))
