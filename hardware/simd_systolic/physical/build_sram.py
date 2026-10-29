#!/usr/bin/env python3
"""Build a pinned ASAP7 6T SRAM research macro with open tools.

Geometry, schematic and estimated timing are distinct evidence. This entry
never marks an uncharacterized macro as physically or frequency certified.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from simd_eval.architecture import ARCH
LOCK=ROOT/'physical/sram_compiler.lock.json'
SOURCE=ROOT/'.deps/openfinram'
PYTHON=ROOT/'.deps/sram-tools/bin/python'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def command(args,log,cwd=ROOT,env=None):
    with Path(log).open('w')as stream:
        subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


def prepare(lock,fetch=False):
    patch=ROOT/lock['patch']
    if sha(patch)!=lock['patch_sha256']:raise ValueError('compiler patch changed')
    if sha(ROOT/'physical/verilog_to_spice.py')!=lock['converter_sha256']:raise ValueError('schematic converter changed')
    if not SOURCE.exists():
        if not fetch:raise ValueError('run with --prepare to fetch the pinned compiler')
        subprocess.run(['git','clone','--no-checkout',lock['repository'],str(SOURCE)],check=True)
        subprocess.run(['git','-C',str(SOURCE),'checkout','--detach',lock['commit']],check=True)
    head=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if head!=lock['commit']:raise ValueError('compiler checkout has a different revision')
    changed=subprocess.check_output(['git','-C',str(SOURCE),'diff','HEAD','--name-only'],text=True).splitlines()
    if not changed:
        subprocess.run(['git','-C',str(SOURCE),'apply',str(patch)],check=True)
    elif set(changed)!=set(lock['patched_files_sha256']):raise ValueError('unexpected compiler source changes')
    for name,digest in lock['patched_files_sha256'].items():
        if sha(SOURCE/name)!=digest:raise ValueError('patched compiler differs: '+name)
    if fetch:subprocess.run(['git','-C',str(SOURCE),'submodule','update','--init','--recursive','--depth','1'],check=True)
    for name,digest in lock['submodules'].items():
        head=subprocess.check_output(['git','-C',str(SOURCE/name),'rev-parse','HEAD'],text=True).strip()
        if head!=digest:raise ValueError('submodule revision differs: '+name)
        if subprocess.check_output(['git','-C',str(SOURCE/name),'diff','HEAD','--name-only'],text=True).strip():
            raise ValueError('submodule source changed: '+name)
    if not PYTHON.exists():
        if not fetch:raise ValueError('run with --prepare to install local KLayout bindings')
        subprocess.run([sys.executable,'-m','venv',str(PYTHON.parents[1])],check=True)
    wheel_root=ROOT/'.deps/sram_wheels';wheel=wheel_root/lock['klayout']['wheel']
    if not wheel.exists():
        if not fetch:raise ValueError('locked KLayout wheel missing')
        subprocess.run([str(PYTHON),'-m','pip','download','--no-deps','klayout=='+lock['klayout']['version'],'--dest',str(wheel_root)],check=True)
    if sha(wheel)!=lock['klayout']['sha256']:raise ValueError('KLayout wheel hash differs')
    installed=subprocess.run([str(PYTHON),'-c','import klayout.db as k;print(k.__version__)'],capture_output=True,text=True)
    if installed.returncode or installed.stdout.strip()!=lock['klayout']['version']:
        subprocess.run([str(PYTHON),'-m','pip','install','--no-deps',str(wheel)],check=True)


def environment():
    env=os.environ.copy();local=ROOT/'.deps/tools/openroad'
    env['PATH']=str(PYTHON.parent)+':'+str(ROOT/'.deps/tools/oss-cad-suite/bin')+':'+env['PATH']
    env['LD_LIBRARY_PATH']=':'.join([str(local/'opt/or-tools/lib'),str(local/'usr/lib/x86_64-linux-gnu'),env.get('LD_LIBRARY_PATH','')])
    env['OPENFINRAM_SPICE_CONVERTER']=str(ROOT/'physical/verilog_to_spice.py')
    return env


def model_check(out,native):
    wrapper=out/'Asap7SramMacro.sv'
    wrapper.write_text(f'''// Logical contract for the generated research macro; no analog timing claim.
`ifdef SYNTHESIS
(* blackbox *)
`endif
module {native}(input clk,ce_n,we_n,oe_n,input [3:0] sdel,
  input [9:0] A,input [63:0] D,output reg [63:0] Q);
`ifndef SYNTHESIS
  reg [63:0] mem [0:1023];
  always @(posedge clk) if(!ce_n) begin
    assert(sdel==0);
    if(!we_n) mem[A]<=D;
    else if(!oe_n) Q<=mem[A];
  end
`endif
endmodule

module srambank_256x4x64_6t122(input clk,input [9:0] ADDRESS,
  input [63:0] wd,input banksel,read,write,output [63:0] dataout);
  {native} storage(.clk(clk),.ce_n(!banksel),.we_n(!write),.oe_n(!read),
    .sdel(4'b0),.A(ADDRESS),.D(wd),.Q(dataout));
endmodule
''')
    build=ROOT/'.build/sram_openfinram_model';build.mkdir(exist_ok=True)
    inputs=[wrapper,ROOT/'rtl/SramBank.sv',ROOT/'tests/sram_bank_main.cpp']
    command(['verilator','--cc','--exe','--build','--assert','-Wno-fatal','-j',4,
        '--top-module','SimdSramBank','-GDEPTH=2048','--Mdir',build/'obj',
        '-o',build/'sim',*inputs],out/'model_build.log')
    result=subprocess.run([str(build/'sim')],check=True,capture_output=True,text=True,timeout=60)
    report=json.loads(result.stdout)
    report.update(inputs_sha256={str(p):sha(p)for p in inputs},physical_certified=False,
        wrapper_sha256=sha(wrapper),bank_adapter_sha256=sha(ROOT/'rtl/SramBank.sv'),
        scope='bank adapter plus generated logical macro model; not transistor simulation')
    (out/'model_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def package(work,run,out,lock):
    native='sram_x256x64x1';tag=run.name.removeprefix(native+'_')
    if out.exists():raise ValueError('output already exists; preserve its evidence and choose another --out')
    required={suffix:run/(native+'.'+suffix)for suffix in ('gds','lef','lib')}
    required['sp']=work/'tmp'/('sram_flat_'+tag+'.sp')
    for path in required.values():
        if not path.is_file()or not path.stat().st_size:raise ValueError('macro view missing: '+str(path))
    out.mkdir(parents=True)
    for suffix,path in required.items():shutil.copy2(path,out/(native+'.'+suffix))
    command([PYTHON,ROOT/'physical/normalize_sram.py',out,native],out/'normalization.log')
    controller=work/'tmp'/('openroad_'+tag)
    for filename in ('openroad.log','run.tcl','timing.rpt','electrical.rpt','detailed_route_drc.rpt',
                     'ctrl_decode.sdc','periphery_implementation.rpt','netlist_for_lvs.conversion.json'):
        if(controller/filename).is_file():shutil.copy2(controller/filename,out/filename)
    command([PYTHON,ROOT/'physical/inspect_sram.py',out,native],out/'inspection.log')
    inspection=json.loads((out/'inspection.json').read_text())
    if inspection['gds_bitcells']!=65536 or inspection['schematic_bitcells']!=65536:
        raise ValueError('generated macro does not retain exactly 1024x64 active bitcells')
    for corner in ('TT','SS','FF'):
        (out/'models').mkdir(exist_ok=True)
        shutil.copy2(SOURCE/f'tech/models/hspice/7nm_{corner}.pm',out/'models'/f'7nm_{corner}.pm')
    model=model_check(out,native)
    report=dict(schema_version=1,kind='generated_asap7_6t_research_macro',native_macro=native,
        architecture_sha256=ARCH['architecture_sha256'],system_macro_count=ARCH['sram_macro_count'],
        system_capacity_bytes=ARCH['sram_capacity_bytes'],
        configuration=lock['configuration'],compiler_commit=lock['commit'],compiler_lock_sha256=sha(LOCK),
        compiler_patch_sha256=lock['patch_sha256'],generator_sha256=sha(__file__),
        converter_sha256=sha(ROOT/'physical/verilog_to_spice.py'),run_directory=str(run),
        read_latency_storage_cycles=1,supported_sdel=0,liberty_kind='estimated',
        functional_model_checked=model['passed'],rtl_wrapper='Asap7SramMacro.sv',
        physical_certified=False,frequency_certified=False,drc_passed=False,lvs_passed=False,
        full_parasitics_extracted=False,pvt_characterized=False,
        inspection=inspection,views_sha256={str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()},
        note='Actual 6T layout and schematic; estimated Liberty is for research only. See failed 2ns timing evidence.')
    (out/'package.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main(args):
    lock=json.loads(LOCK.read_text());prepare(lock,args.prepare)
    cfg=lock['configuration']
    if (cfg['logical_depth'],cfg['logical_width'],cfg['memory_period_ps'])!=(ARCH['sram_macro_depth'],ARCH['sram_macro_width'],ARCH['memory_period_ps']):
        raise ValueError('compiler configuration differs from architecture.properties')
    work=args.work.resolve();work.mkdir(parents=True,exist_ok=True)
    tech=work/'tech'
    if not tech.exists():tech.symlink_to(SOURCE/'tech',target_is_directory=True)
    if tech.resolve()!=(SOURCE/'tech').resolve():raise ValueError('work directory has another technology source')
    if args.reuse_run:
        run=args.reuse_run.resolve()
        if run.parent!=(work/'results').resolve():raise ValueError('reuse run must belong to this compiler work directory')
    else:
        build=ROOT/'.build/openfinram'
        command(['cmake','-S',SOURCE,'-B',build,'-DCMAKE_BUILD_TYPE=Release'],work/'configure.log')
        command(['cmake','--build',build,'-j',args.jobs],work/'build.log')
        command_line=[build/'OpenFinRAM','--num-wls',128,'--num-data-bits',64,'--num-banks',1,
            '--single-port','--num-rows-per-mux',4,'--openroad','--openroad-path',
            ROOT/'.deps/tools/openroad/usr/bin/openroad','--platform-path',SOURCE/'tech','--skip-characterization']
        if args.emit_unverified_layout:command_line+=['--emit-unverified-layout']
        log=work/('run_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')+'.log')
        command(command_line,log,cwd=work,env=environment())
        run=max((work/'results').glob('sram_x256x64x1_*'),key=lambda p:p.stat().st_mtime)
    report=package(work,run,args.out.resolve(),lock)
    print(json.dumps({k:report[k]for k in ('kind','native_macro','physical_certified','frequency_certified')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    p.add_argument('--emit-unverified-layout',action='store_true')
    p.add_argument('--work',type=Path,default=ROOT/'.build/sram_openfinram_1024x64')
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/migration_asap7_f2/sram_openfinram')
    p.add_argument('--reuse-run',type=Path);p.add_argument('--jobs',type=int,default=8)
    main(p.parse_args())
