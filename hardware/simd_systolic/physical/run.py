#!/usr/bin/env python3
"""Reproducible synthesis and extracted timing; absent evidence never certifies a target."""
import argparse,fcntl,hashlib,json,os,re,shutil,signal,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
DRIVER_SOURCE=Path(__file__).read_bytes()
sys.path.insert(0,str(ROOT/'python'))
from simd_eval.constraints import core_policy, system_policy
from simd_eval.architecture import ARCH, require_f2
from asap7_platform import profile, excluded_cells, groups, block_end
from build import sources,rtl_digest,elaborate
from runtime import validated_build
from memory_inventory import inventory as memory_inventory
from sram_package import load_package,inventory as generated_memory_inventory
from pvt import matching_points
from coverage import analyze as analyze_coverage

def sha(path):
    # Full-array checkpoints exceed 10 GB; hashing must not allocate another
    # checkpoint-sized buffer while physical implementation is resident.
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def tool(name):
    local=ROOT/'.deps/tools'
    for p in [local/'oss-cad-suite/bin'/name,local/'openroad/usr/bin'/name]:
        if p.exists():return str(p)
    found=shutil.which(name)
    if not found:raise RuntimeError(f'{name} unavailable; run scripts/bootstrap.py --physical')
    return found

def int64_component(inputs,out):
    """Specialize the actual generated ALU to I64, keeping its two-stage path.

    The source ALU is copied and renamed. Only the format input is constant;
    opcodes, masks, 144-bit accumulation, conversion, reset and stalls stay live.
    The new RTL identity must not be relabeled as the parent full-format ALU.
    """
    original=[p for p in inputs if p.name=='SimdAlu.sv']
    if len(original)!=1:raise ValueError('I64 component needs one generated ALU')
    text=original[0].read_text()
    if text.count('module SimdAlu(')!=1:raise ValueError('unexpected generated ALU declaration')
    renamed=out/'SimdAluGeneric.sv';renamed.write_text(text.replace('module SimdAlu(','module SimdAluGeneric(',1))
    wrapper=out/'SimdInt64.sv'
    wrapper.write_text('''module SimdInt64(
 input clock,reset,io_in_valid,io_out_ready,
 input [63:0] io_in_bits_a,io_in_bits_b,
 input [511:0] io_in_bits_acc,
 input [2:0] io_in_bits_op,
 input [15:0] io_in_bits_mask,
 input [3:0] io_in_bits_ctx,
 input [31:0] io_in_bits_tag,
 output io_in_ready,io_out_valid,
 output [511:0] io_out_bits_data,
 output [4:0] io_out_bits_flags,
 output [15:0] io_out_bits_overflow,
 output io_out_bits_illegal,
 output [3:0] io_out_bits_ctx,
 output [31:0] io_out_bits_tag);
 SimdAluGeneric arithmetic(.io_in_bits_fmt(4'd4),.*);
endmodule
''')
    return [renamed if p==original[0]else p for p in inputs]+[wrapper]

def mapping_library(out,lib):
    """Apply the pinned platform's dont-use policy during ABC mapping too."""
    raw=lib.read_text();removed=excluded_cells(raw)
    for name,begin,end in reversed(list(groups(raw,r'\bcell'))):
        if name in removed:raw=raw[:begin]+raw[end:]
    filtered=out/'mapping.lib';filtered.write_text(raw)
    return filtered,removed

def synthesize(out,top,inputs,lib,period,env,pe_hierarchy=False,mapping_profile='full',sequential_profile='standard'):
    ys=out/'elaborate.ys'
    # Slang flattens instances before Yosys can honor keep_hierarchy. Import
    # this small, fixed-interface register primitive through the native
    # frontend, so its kept local instances survive both synthesis stages.
    local_controls=[p for p in inputs if p.name in('ControlReplicas.sv','OperandReplicas.sv','PopulationCount.sv')]
    prefix=''.join(f'read_verilog -sv {p}\n'for p in local_controls)
    def read_slang(name,files):
        return 'read_slang --ignore-assertions -DSYNTHESIS -I'+str(out/'inputs')+' --top '+name+' '+' '.join(map(str,files))+'\n'
    files=[p for p in inputs if p not in local_controls]
    if pe_hierarchy:
        # Elaborate one actual array PE, then import it through its exact
        # interface into the mesh. This preserves cell names for 2x2 fences
        # and maps the common PE once instead of flattening 64 arithmetic trees.
        prefix+=read_slang('SimdPE',[p for p in files if p.name!='SystolicArray.sv'])
        prefix+='setattr -mod -set keep_hierarchy 1 SimdPE\n'
        prefix+=read_slang(top,[p for p in files if p.name!='SimdPE.sv'])
    else:prefix+=read_slang(top,files)
    # Padded request words can have identical constant payloads. Preserve
    # their distinct local dependency controllers rather than allowing
    # opt_merge to recreate the measured global capture fanout.
    prefix+='setattr -set keep 1 t:SimdElasticPreparationWord\n'
    ys.write_text(prefix+f'hierarchy -check -top {top}\nsynth -top {top} -flatten -noabc -noshare\nwrite_rtlil -readable {out/"premap.il"}\n')
    with(out/'elaborate.log').open('w')as log:
        subprocess.run([tool('yosys'),'-Q','-T','-s',str(ys)],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    identity_path=out/'build_identity.json'
    identity=json.loads(identity_path.read_text());identity['premap_sha256']=sha(out/'premap.il')
    identity_path.write_text(json.dumps(identity,indent=2)+'\n')
    map_elaborated(out,lib,period,env,mapping_profile,sequential_profile)

def map_elaborated(out,lib,period,env,mapping_profile='full',sequential_profile='standard'):
    # $scopeinfo is metadata introduced by current Yosys flattening; Yosys
    # 0.33 cannot import it. Remove only that metadata in the current frontend.
    compat=out/'compat.ys'
    compat.write_text(f'read_rtlil {out/"premap.il"}\ndelete t:$scopeinfo\nwrite_rtlil -readable {out/"premap_compat.il"}\n')
    with(out/'compat.log').open('w')as log:
        subprocess.run([tool('yosys'),'-Q','-T','-s',str(compat)],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    mapper=shutil.which('yosys')
    if not mapper or Path(mapper).resolve()==Path(tool('yosys')).resolve():
        raise RuntimeError('stable system Yosys mapper required (tested Ubuntu Yosys 0.33)')
    abc_lib,excluded=mapping_library(out,lib)
    from async_mapping_policy import qualify as qualify_async
    async_metadata=qualify_async(out,lib)
    sequential_commands=f'''dffunmap
techmap -map {out/'Asap7AsyncFlops.v'}
techmap
select -assert-none t:$_DFFSR_* t:$_DFF_*0_ t:$_DFF_*1_ t:$_DFFE_*0?_ t:$_DFFE_*1?_
'''
    sequential_metadata={'sequential_mapping_profile':sequential_profile}
    if sequential_profile!='standard':raise ValueError('only ASAP7 standard sequential mapping is supported')
    macro_lib=env.get('SIMD_MACRO_LIB',str(ROOT/'.deps/asap7_sram/generated/LIB/srambank_256x4x64_6t122.lib'))
    abc_options='-fast 'if mapping_profile=='fast'else''
    if mapping_profile=='choices':
        # Combinational structural choices without the expensive full-flow
        # fraiging/scorr/dc2 search. Registers never enter ABC (-dff is absent).
        policy=out/'mapping_choices.abc'
        policy.write_text(f'strash; &get -n; &dch -f; &nf -D {period*1000-150:g}; &put\n')
        abc_options=f'-script {policy} '
        sequential_metadata.update(abc_script_sha256=sha(policy),mapping_timeout_seconds=600)
    elif mapping_profile not in('fast','full'):raise ValueError('unsupported ABC mapping profile')
    ys=out/'synth.ys';ys.write_text(f'''read_liberty -lib {lib}
read_liberty -lib {macro_lib}
read_rtlil {out/'premap_compat.il'}
{sequential_commands}dfflibmap -liberty {lib}
abc {abc_options}-liberty {abc_lib} -D {period*1000-150:g}
setundef -zero
hilomap -singleton -hicell TIEHIx1_ASAP7_75t_R H -locell TIELOx1_ASAP7_75t_R L
clean
check -assert
stat -liberty {lib}
tee -o {out/'cell_statistics.json'} stat -json -liberty {lib}
write_verilog -noattr -noexpr {out/'synth.v'}
''')
    with(out/'synth.log').open('w')as log:
        command=[mapper,'-Q','-T','-s',str(ys)]
        if mapping_profile=='choices':
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:code=process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                # This group was created here and contains only this mapper
                # and its ABC children. Never terminate unrelated tool jobs.
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
                # The parent can exit before a child handles SIGTERM.
                # Terminate any remaining members of this same private group.
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()
                (out/'mapping_failure.json').write_text(json.dumps(dict(
                    status='bounded_mapping_timeout',timeout_seconds=600,
                    script_sha256=sha(out/'mapping_choices.abc'),synthesis_script_sha256=sha(ys)))+'\n')
                raise
            if code:raise subprocess.CalledProcessError(code,command)
        else:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    excluded_set=set(excluded)
    with(out/'synth.v').open()as mapped:
        for line in mapped:
            head=line.split(None,1)
            if head and head[0]in excluded_set:
                raise RuntimeError('mapping ignored dont-use cell policy: '+head[0])
    identity_path=out/'build_identity.json';identity=json.loads(identity_path.read_text())
    identity.update(abc_library_sha256=sha(abc_lib),excluded_mapping_cells=excluded,mapping_profile=mapping_profile,**sequential_metadata,**async_metadata)
    identity_path.write_text(json.dumps(identity,indent=2)+'\n')

def mapped_area_breakdown(standard_area, macro_count=0, macro_lef=None, macro_name=None):
    """Yosys stat reads standard cells only; use the 1x LEF macro footprint."""
    macro_area=0.0;evidence={}
    if macro_count:
        raw=Path(macro_lef).read_text();name=re.escape(macro_name or ARCH['sram_macro'])
        cell=re.search(r'\bMACRO\s+'+name+r'\b(.*?)\bEND\s+'+name+r'\b',raw,re.S)
        match=re.search(r'\bSIZE\s+([\d.eE+-]+)\s+BY\s+([\d.eE+-]+)\s*;',cell[1])if cell else None
        if not match:raise ValueError('selected SRAM cell has no LEF footprint')
        macro_area=macro_count*float(match[1])*float(match[2])
        evidence=dict(sram_area_lef=str(Path(macro_lef).resolve()),sram_area_lef_sha256=sha(macro_lef))
    return dict(standard_cell_area_um2=standard_area,sram_macro_area_um2=macro_area,
        mapped_cell_and_macro_area_um2=None if standard_area is None else standard_area+macro_area,
        floorplan_area_um2=None,area_evidence='mapped standard cells plus candidate 1x LEF footprint; no floorplan',**evidence)


def summarize(out,identity,period):
    from async_mapping_policy import validate as validate_async
    async_validation=validate_async(out,identity,out/'routed.v')
    identity={**identity,'asynchronous_mapping_validation':async_validation}
    if identity['scope'] in ('system','gemm'):
        if sha(Path(identity['sram_liberty_file']))!=identity['sram_liberty_sha256']:
            raise ValueError('SRAM Liberty changed after implementation')
    log=(out/'route.log').read_text()
    def slack(marker):
        m=re.search(marker+r'.*?worst slack\s+([-+\d.]+)',log,re.S)
        if not m:raise RuntimeError('missing timing marker '+marker)
        return float(m[1])
    setup,hold=slack('SIMD_FINAL_SETUP'),slack('SIMD_FINAL_HOLD')
    tns=float(re.search(r'SIMD_FINAL_TNS.*?tns\s+([-+\d.]+)',log,re.S)[1])
    from parasitics import audit as audit_parasitics
    driver_audit=out/'parasitic_driver_audit.json'
    parasitics=audit_parasitics(log,out/'design.spef',driver_audit if driver_audit.exists()else None)
    parasitics['spef_sha256']=sha(out/'design.spef')
    (out/'parasitic_annotation.json').write_text(json.dumps(parasitics,indent=2)+'\n')
    areas=re.findall(r'Design area\s+([\d.]+)',log)
    evidence={name:str(out/file)for name,file in [('netlist','routed.v'),('sdc','design.sdc'),
      ('spef','design.spef'),('setup','setup.rpt'),('hold','hold.rpt'),('electrical','electrical.rpt'),
      ('coverage','coverage.rpt'),('drc','route.drc')]}
    evidence['parasitic_annotation']=str(out/'parasitic_annotation.json')
    if driver_audit.exists():evidence['parasitic_driver_audit']=str(driver_audit)
    if identity.get('finalization_driver_sha256'):
        path=out/'finalization_driver.py'
        if sha(path)!=identity['finalization_driver_sha256']:raise ValueError('finalization driver snapshot changed')
        evidence['finalization_driver']=str(path)
    if identity.get('canonical_export_manifest_sha256'):
        path=out/'canonical_export.json'
        if sha(path)!=identity['canonical_export_manifest_sha256']:raise ValueError('canonical export manifest changed')
        export=json.loads(path.read_text())
        if not export['geometry_and_connectivity_preserved']:raise ValueError('canonical export geometry differs')
        for filename,digest in export['evidence'].items():
            if sha(filename)!=digest:raise ValueError('canonical export evidence changed: '+filename)
        driver=out/'recovery_driver.py'
        if sha(driver)!=identity['export_recovery_driver_sha256']:raise ValueError('export recovery driver changed')
        evidence.update(canonical_export=str(path),export_recovery_driver=str(driver))
    if identity.get('physical_driver_sha256'):
        if sha(out/'physical_driver.py')!=identity['physical_driver_sha256']:
            raise ValueError('executed physical driver snapshot changed')
        evidence['physical_driver']=str(out/'physical_driver.py')
    if identity.get('mapping_profile')=='choices':
        if sha(out/'mapping_choices.abc')!=identity['abc_script_sha256']:
            raise ValueError('ABC mapping policy snapshot changed')
        evidence['abc_policy']=str(out/'mapping_choices.abc')
    for path in evidence.values():
        if not Path(path).exists():raise RuntimeError('missing signoff evidence: '+path)
    electrical=(out/'electrical.rpt').read_text();coverage=(out/'coverage.rpt').read_text()
    drc=(out/'route.drc').read_text()
    electrical_clean='VIOLATED'not in electrical.upper()
    process_library=Path(identity['liberty_file'])
    if sha(process_library)!=identity['liberty_sha256']:raise RuntimeError('coverage process library changed')
    coverage_analysis=analyze_coverage((out/'routed.v').read_text(),coverage,process_library.read_text())
    coverage_analysis['liberty_sha256']=identity['liberty_sha256']
    coverage_analysis['netlist_sha256']=sha(out/'routed.v')
    coverage_file=out/'coverage_analysis.json'
    coverage_file.write_text(json.dumps(coverage_analysis,indent=2)+'\n')
    evidence['coverage_analysis']=str(coverage_file)
    coverage_clean=coverage_analysis['clean']
    drc_clean=not drc.strip()
    # Core reset ports are synchronous D-path controls (RequireSyncReset).
    # They must not inherit an asynchronous reset-pin timing exception.
    reset_present=identity['scope']!='fp64'
    reset_waived=reset_present and bool(re.search(r'set_false_path[^\n]*\breset\b',(out/'design.sdc').read_text()))
    if identity['scope'] in ('system','gemm'):
        from mapped_system_constraints import system_sdc
        for name,filename,key in [('cdc_helper','cdc_constraints.tcl','cdc_helper_sha256'),
                                 ('cdc_generator','mapped_system_constraints.py','mapped_constraint_generator_sha256'),
                                 ('macro_floorplan','sram_floorplan.tcl','macro_floorplan_sha256'),
                                 ('protected_power','protected_power.tcl','protected_power_sha256'),
                                 ('power_grid_strategy','pdn.tcl','power_grid_strategy_sha256')]:
            path=out/filename
            if not path.is_file()or identity.get(key)!=sha(path):raise ValueError('CDC constraint source snapshot changed: '+filename)
            evidence[name]=str(path)
        constraint_analysis=system_policy((out/'design.sdc').read_text(),system_sdc(out/'synth.v',out/'cdc_constraints.tcl'))
    else:
        constraint_analysis=core_policy((out/'design.sdc').read_text(),period)
    constraint_analysis['sdc_sha256']=sha(out/'design.sdc')
    constraint_file=out/'constraint_analysis.json'
    constraint_file.write_text(json.dumps(constraint_analysis,indent=2)+'\n')
    evidence['constraint_analysis']=str(constraint_file)
    constraint_clean=not reset_waived and constraint_analysis['clean']
    macro_clean=identity['scope'] not in ('system','gemm') or (identity.get('sram_physical_certified',False) and identity.get('sram_corner_matched',False))
    power_clean='SIMD_FINAL_POWER_GRID_VERIFIED'in log
    closed=parasitics['clean'] and period==ARCH['core_period_ps']/1000 and setup>=0 and hold>=0 and tns==0 and electrical_clean and coverage_clean and drc_clean and constraint_clean and macro_clean and power_clean and identity.get('process')=='asap7' and identity.get('architecture_sha256')==ARCH['architecture_sha256'] and not identity.get('provenance_status','').startswith('historical')
    scope=identity['scope'];rows=identity['rows']
    report={**identity,'period_ns':period,'setup_wns_ns':setup,'hold_wns_ns':hold,'tns_ns':tns,
      'timing_status':'array_closed'if closed and scope=='array'and rows==8 else'kernel_closed'if closed else'not_closed',
      'setup_hold_closed':parasitics['clean'] and setup>=0 and hold>=0 and tns==0,
      'parasitic_annotation_clean':parasitics['clean'],
      'electrical_clean':electrical_clean,'coverage_clean':coverage_clean,'route_drc_clean':drc_clean,
      'power_grid_connectivity_verified':power_clean,
      'constraint_policy_clean':constraint_clean,'constraint_policy_version':2,'synchronous_reset_present':reset_present,'synchronous_reset_timed':not reset_waived,
      'constraint_limitations':['synchronous reset input was false-pathed']if reset_waived else[],
      'reference_process':'ASAP7 predictive 7nm RVT; research PDK' if identity.get('process')=='asap7' else 'historical process; inspect authenticated Liberty',
      'timing_report_decimal_places_ns':9,
      'cell_area_um2':float(areas[-1])if areas else None,'netlist_sha256':sha(out/'routed.v'),
      'sdc_sha256':sha(out/'design.sdc'),'flow_sha256':sha(out/'flow.tcl'),
      'evidence':evidence,'evidence_sha256':{name:sha(path)for name,path in evidence.items()}}
    (out/'timing.json').write_text(json.dumps(report,indent=2)+'\n');return report

def run(args):
    target,stages,rows,period=args.target,args.stages,args.rows,args.period
    if sum((args.resume_cts,args.resume_initial_hold,args.resume_global_route,args.resume_detailed_route))>1 or (args.cts_only and(args.resume_initial_hold or args.resume_global_route or args.resume_detailed_route)) or (args.detailed_route_only and(args.cts_only or args.resume_detailed_route)):
        raise ValueError('choose one physical resume stage; post-CTS stages cannot be CTS-only')
    require_f2(stages)
    technology=profile(args.corner)
    if args.rtl_build and (args.map_from or args.reuse_mapped or args.remap_premap):
        raise ValueError('choose an RTL snapshot or an existing mapping, not both')
    if period!=ARCH['core_period_ps']/1000:
        raise ValueError('ASAP7 F2 acceptance fixes the core period at 1ns; do not lower the target frequency')
    env=os.environ.copy();local=ROOT/'.deps/tools/openroad';pdk=technology['root']
    env['LD_LIBRARY_PATH']=':'.join([str(local/'opt/or-tools/lib'),str(local/'usr/lib/x86_64-linux-gnu'),env.get('LD_LIBRARY_PATH','')])
    label=('_'+args.label)if args.label else''
    out=ROOT/f'.build/physical/asap7_{args.corner}_{target}_{stages}_{rows}x{rows}_{period:g}ns{label}'
    out.mkdir(parents=True,exist_ok=True);guard=(out/'.lock').open('w');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    driver=out/'physical_driver.py'
    if driver.exists()and driver.read_bytes()!=DRIVER_SOURCE:
        history=out/('physical_driver_'+sha(driver)[:12]+'.py')
        if history.exists()and history.read_bytes()!=driver.read_bytes():raise ValueError('physical driver archive collision')
        if not history.exists():shutil.copy2(driver,history)
    driver.write_bytes(DRIVER_SOURCE)
    macro_package=None
    if args.sram_package:
        if target not in('system','gemm'):raise ValueError('SRAM packages apply only to integrated targets')
        if args.rtl_build:raise ValueError('a macro replacement cannot relabel an existing frozen RTL snapshot; elaborate its actual wrapper')
        macro_package=load_package(args.sram_package,ARCH)
    if target in ('system','gemm'):
        from sram_signoff import audit
        macro_audit=(dict(macro=macro_package['native'],physical_certified=False,corner_libraries={},
            package_manifest_sha256=macro_package['manifest_sha256'],
            gaps=['Generated research macro has estimated timing, incomplete DRC/LVS/PEX and PVT characterization'])
            if macro_package else audit())
        selected_macro_lib=(macro_package['liberty']if macro_package else Path(macro_audit['corner_libraries'].get(args.corner,ROOT/'.deps/asap7_sram/generated/LIB/srambank_256x4x64_6t122.lib')))
        selected_macro_lef=(macro_package['lef']if macro_package else ROOT/'.deps/asap7_sram/generated/LEF/srambank_256x4x64_6t122.lef')
        env['SIMD_MACRO_LIB']=str(selected_macro_lib)
        (out/'sram_audit.json').write_text(json.dumps(macro_audit,indent=2)+'\n')
        if not args.synth_only and not args.allow_unverified_sram and not macro_audit['physical_certified']:
            raise ValueError('SRAM physical prerequisites missing; see '+str(out/'sram_audit.json'))
    identity_file=out/'build_identity.json';lib=technology['liberty']
    if args.map_from:
        source=args.map_from.resolve();identity=json.loads((source/'build_identity.json').read_text())
        if identity.get('architecture_sha256')!=ARCH['architecture_sha256'] or identity.get('process_corner')!=args.corner:raise ValueError('cannot reuse another process/architecture/corner')
        if (identity['scope'],identity['stages'],identity['rows'])!=(target,stages,rows):
            raise ValueError('mapped implementation has different scope/stages/dimensions')
        shutil.copytree(source/'inputs',out/'inputs',dirs_exist_ok=True)
        shutil.copy2(source/'synth.v',out/'synth.v')
        if(source/'generated/capabilities.json').exists():
            (out/'generated').mkdir(exist_ok=True)
            shutil.copy2(source/'generated/capabilities.json',out/'generated/capabilities.json')
        for name in('cell_statistics.json','synth.log','mapping.lib','Asap7AsyncFlops.v','NangateSyncControls.v','sequential_mapping_proof.json'):
            if(source/name).exists():shutil.copy2(source/name,out/name)
        identity['mapped_from']=str(source)
    elif args.reuse_mapped:
        identity=json.loads(identity_file.read_text())
        if identity.get('architecture_sha256')!=ARCH['architecture_sha256'] or identity.get('process_corner')!=args.corner:raise ValueError('cannot reuse historical mapping')
        if(identity['scope'],identity['stages'],identity['rows'])!=(target,stages,rows):raise ValueError('mapped implementation has different scope/stages/dimensions')
    elif args.remap_premap:
        identity=json.loads(identity_file.read_text())
        if (identity['scope'],identity['stages'],identity['rows'])!=(target,stages,rows):raise ValueError('elaboration identity mismatch')
        if sha(out/'premap.il')!=identity['premap_sha256']:raise RuntimeError('elaborated netlist changed')
        if rtl_digest(list((out/'inputs').glob('*.sv')),out/'inputs/common_cells')!=identity['rtl_sha256']:
            raise RuntimeError('elaboration source snapshot changed')
        map_elaborated(out,lib,period,env,args.mapping_profile or identity.get('mapping_profile','full'),
            args.sequential_profile or identity.get('sequential_mapping_profile','standard'))
        identity=json.loads(identity_file.read_text())
        identity['mapped_sha256']=sha(out/'synth.v')
    else:
        if target=='fp64':
            top='SimdFpF264'if stages==2 else'SimdFpHF64'if stages==8 else'SimdFpHF12_64';rtl=[]
            if args.free_running:
                unit=top;top='SimdFpCore64'
                wrapper=out/'fp64_core.sv'
                wrapper.write_text(f'''module {top}(input clock,input [63:0] a,b,c,input [2:0] op,
output [63:0] result,output [4:0] flags);
{unit} unit(.clock(clock),.ce(1'b1),.a(a),.b(b),.c(c),.op(op),.result(result),.flags(flags));
endmodule
''')
                rtl=[wrapper]
        else:
            if args.free_running:raise ValueError('free-running wrapper applies only to the FP64 arithmetic probe')
            top={'alu':'SimdAlu','int64':'SimdInt64','pe':'SimdPE','array':'SystolicArray','system':'TileSystem','gemm':'GemmSystem'}[target]
            gen=out/'generated'
            if args.rtl_build:
                source_build=args.rtl_build.resolve()
                source_guard,source_caps=validated_build(source_build)
                parent_target='alu'if target=='int64'else target
                if(source_caps['target'],source_caps['stages'],source_caps['rows'],source_caps['cols'])!=(parent_target,stages,rows,rows):
                    raise ValueError('verified RTL build scope/stages/dimensions differ')
                gen.mkdir(exist_ok=True)
                shutil.copy2(source_build/'capabilities.json',gen/'capabilities.json')
                for p in source_build.glob('*.sv'):shutil.copy2(p,gen/p.name)
                rtl=sorted((source_build/'inputs').glob('*.sv'))
            else:
                elaborate('alu'if target=='int64'else target,stages,rows,rows,gen)
                rtl=sorted(gen.glob('*.sv'))
        if args.rtl_build and target=='fp64':raise ValueError('RTL build snapshots require a complete generated target')
        original=rtl if args.rtl_build else sources()+rtl
        if target=='int64':original=int64_component(original,out)
        if macro_package:original=[macro_package['wrapper']if p.name=='Asap7SramMacro.sv'else p for p in original]
        headers=source_build/'inputs/common_cells'if args.rtl_build else ROOT/'vendor/common_cells'
        snapshot=out/'inputs';snapshot.mkdir(exist_ok=True)
        inputs=[]
        for p in original:
            q=snapshot/p.name;q.write_bytes(p.read_bytes());inputs.append(q)
        shutil.copytree(headers,snapshot/'common_cells',dirs_exist_ok=True)
        identity={'scope':target,'top':top,'stages':stages,'rows':rows,'cols':rows,
          'rtl_sha256':rtl_digest(original,headers),'liberty_sha256':sha(lib),'mapping_period_ns':period,
          'input_manifest':{p.name:sha(p)for p in inputs},'pe_hierarchy':args.pe_hierarchy}
        if macro_package:identity['sram_package_manifest_sha256']=macro_package['manifest_sha256']
        if target=='fp64':identity['fp64_free_running']=args.free_running
        if args.rtl_build:
            if target=='int64':identity['parent_alu_rtl_sha256']=source_caps['rtl_sha256']
            elif identity['rtl_sha256']!=source_caps['rtl_sha256']:raise RuntimeError('verified RTL snapshot digest changed')
            identity.update(qualified_rtl_build=str(source_build),qualified_simulator_sha256=source_caps['simulator_sha256'])
        if target=='int64':identity.update(component_format='I64',component_format_code=4,
            provenance_status='specialized_component_only',full_format_alu_certified=False)
        if args.pe_hierarchy and target not in('array','system','gemm'):raise ValueError('PE hierarchy requires an array or integrated system')
        identity_file.write_text(json.dumps(identity,indent=2)+'\n')
        synthesize(out,top,inputs,lib,period,env,args.pe_hierarchy,args.mapping_profile or'full',args.sequential_profile or'standard')
        identity=json.loads(identity_file.read_text())
        identity['premap_sha256']=sha(out/'premap.il')
        identity['mapped_sha256']=sha(out/'synth.v')
    if sha(out/'synth.v')!=identity['mapped_sha256']or sha(lib)!=identity['liberty_sha256']:
        raise RuntimeError('mapped netlist/library identity changed')
    from async_mapping_policy import validate as validate_async
    identity['asynchronous_mapping_validation']=validate_async(out,identity)
    package_digest=macro_package['manifest_sha256']if macro_package else None
    if (args.map_from or args.reuse_mapped or args.remap_premap)and identity.get('sram_package_manifest_sha256')!=package_digest:
        raise ValueError('mapped implementation has a different SRAM package identity')
    if args.sequential_profile and args.sequential_profile!=identity.get('sequential_mapping_profile','standard'):
        raise ValueError('sequential profile differs from the existing mapping; remap rather than re-label it')
    if identity.get('sequential_mapping_profile')=='scan_controls':
        for name,key in [('NangateSyncControls.v','sequential_mapping_sha256'),
                         ('sequential_mapping_proof.json','sequential_mapping_proof_sha256')]:
            if sha(out/name)!=identity[key]:raise ValueError('frozen sequential mapping evidence changed: '+name)
    if identity.get('abc_library_sha256')and sha(out/'mapping.lib')!=identity['abc_library_sha256']:
        raise RuntimeError('mapping library snapshot changed')
    identity.update(architecture_sha256=ARCH['architecture_sha256'],process='asap7',process_corner=args.corner,core_period_ps=ARCH['core_period_ps'],memory_period_ps=ARCH['memory_period_ps'],sram_physical_certified=macro_audit['physical_certified'] if target in ('system','gemm') else False)
    identity['physical_driver_sha256']=hashlib.sha256(DRIVER_SOURCE).hexdigest()
    identity['analysis_period_ns']=period
    identity['liberty_file']=str(lib)
    if target in ('system','gemm'):
        identity['sram_liberty_file']=str(selected_macro_lib)
        identity['sram_liberty_sha256']=sha(selected_macro_lib)
        identity['sram_pvt_comparison']=matching_points(lib,selected_macro_lib)
        identity['sram_corner_matched']=(args.corner in macro_audit['corner_libraries'] and
            identity['sram_pvt_comparison']['matched'])
        identity['sram_macro_name']=macro_package['native']if macro_package else ARCH['sram_macro']
        identity['sram_package_manifest_sha256']=package_digest
        identity['sram_lef_file']=str(selected_macro_lef)
        identity['sram_lef_sha256']=sha(selected_macro_lef)
    if args.placement_profile:identity['placement_profile']=args.placement_profile
    identity.setdefault('placement_profile','timing')
    if args.placement_grid is not None:
        grid=args.placement_grid
        if grid<32 or grid>2048 or grid&(grid-1):raise ValueError('placement grid must be a power of two in [32,2048]')
        identity['placement_grid_count']=grid
    identity.setdefault('routed_hold_repair_margin_ns',.08 if target=='fp64'else .05)
    identity.setdefault('setup_repair_margin_ns',.05)
    if args.setup_margin is not None:
        if not 0<=args.setup_margin<=.5:raise ValueError('setup margin must be in [0,0.5] ns')
        identity['setup_repair_margin_ns']=args.setup_margin
    if args.hold_margin is not None:
        if not 0<=args.hold_margin<=.5:raise ValueError('hold repair margin must be in [0,0.5] ns')
        identity['routed_hold_repair_margin_ns']=args.hold_margin
    if args.clock_profile:
        if args.resume_cts or args.resume_initial_hold or args.resume_global_route or args.resume_detailed_route:raise ValueError('a resumed checkpoint retains its existing clock tree')
        identity['clock_tree_options']=(dict(root=technology['cts_root'],buffers=technology['cts_buffers'],
          sink_clustering=True,cluster_size=64 if args.clock_profile=='clustered64'else 20,cluster_diameter_um=50)
          if args.clock_profile in('clustered','clustered64') else
          dict(root=technology['cts_root'],buffers=technology['cts_buffers'],sink_clustering=False))
    identity_file.write_text(json.dumps(identity,indent=2)+'\n')
    if (out/'synth.log').exists():
        stats=out/'cell_statistics.json'
        area=None;replicas=None;source_replicas=None;operation_replicas=None;wave_replicas=None;context_replicas=None
        observation_replicas=None;observation_state_replicas=None;observation_state_flops=None;write_replicas=None
        commit_replicas=None;seed_replicas=None;valid_replicas=None;preparation_replicas=None;packed_a_replicas=None
        elastic_word_replicas=None
        if stats.exists():
            data=json.loads(stats.read_text());area=data['design']['area']
            modules={k.lstrip('\\'):v for k,v in data['modules'].items()}
            def count(name,wanted):
                return sum(n if typ.lstrip('\\')==wanted else n*count(typ.lstrip('\\'),wanted)
                    for typ,n in modules.get(name,{}).get('num_cells_by_type',{}).items())
            replicas=count(identity['top'],'SimdControlRegister')
            source_replicas=count(identity['top'],'SimdSourceFormatDecode')
            operation_replicas=count(identity['top'],'SimdOpControlRegister')
            wave_replicas=count(identity['top'],'SimdWaveRegister')
            context_replicas=count(identity['top'],'SimdContextReadRegister')
            observation_replicas=count(identity['top'],'SimdObservationSelect')+count(identity['top'],'SimdObservationState')
            observation_state_replicas=count(identity['top'],'SimdObservationState')
            observation_state_flops=sum(n for typ,n in modules.get('SimdObservationState',{}).get('num_cells_by_type',{}).items()
                if typ.lstrip('\\').startswith(('DFF','SDFF')))
            write_replicas=count(identity['top'],'SimdStateWriteSelect')
            commit_replicas=count(identity['top'],'SimdPeCommitWord')+count(identity['top'],'SimdPeElasticCommitWord')
            seed_replicas=count(identity['top'],'SimdPeSeedWord')
            valid_replicas=count(identity['top'],'SimdValidRegister')
            elastic_word_replicas=count(identity['top'],'SimdElasticPreparationWord')
            preparation_replicas=count(identity['top'],'SimdPreparationControlRegister')+count(identity['top'],'SimdElasticPreparationControlRegister')+elastic_word_replicas
            packed_a_replicas=count(identity['top'],'SimdPackedARegister')
        else:
            matches=re.findall(r"Chip area for (top )?module '([^']+)':\s*([\d.]+)",(out/'synth.log').read_text())
            preferred=[float(a)for kind,name,a in matches if kind]
            if preferred:area=preferred[-1]
            else:
                matching=[float(a)for _,name,a in matches if name.lstrip('\\')==identity['top']]
                if matching:area=matching[-1]
        synthesis={**identity,'evidence_kind':'standard_cell_synthesis',
          'cell_area_um2':area,'control_replica_instances':replicas,'source_format_replica_instances':source_replicas,
          'operation_replica_instances':operation_replicas,'timing_status':'synthesis_only',
          'wave_register_instances':wave_replicas,
          'context_read_replica_instances':context_replicas,
          'observation_select_instances':observation_replicas,'state_write_select_instances':write_replicas,
          'observation_state_instances':observation_state_replicas,
          'observation_initialized_flipflops_per_instance':observation_state_flops,
          'commit_word_instances':commit_replicas,
          'seed_word_instances':seed_replicas,
          'response_valid_replica_instances':valid_replicas,
          'preparation_control_replica_instances':preparation_replicas,
          'elastic_preparation_word_instances':elastic_word_replicas,
          'packed_a_register_instances':packed_a_replicas,
          'netlist':str(out/'synth.v'),'report':str(out/'synth.log')}
        if target in('system','gemm'):
            memory=(generated_memory_inventory(out/'synth.v',macro_package['native'])if macro_package else memory_inventory(out/'synth.v'))
            (out/'memory_inventory.json').write_text(json.dumps(memory,indent=2)+'\n')
            synthesis['memory_inventory']=memory
            capabilities=out/'generated/capabilities.json'
            if capabilities.exists()and json.loads(capabilities.read_text()).get('scratchpad_access')and not memory['all_nominal_bits_retained']:
                raise RuntimeError('accessible scratch banks were optimized below their declared capacity; inspect memory_inventory.json')
        synthesis.update(mapped_area_breakdown(area,16 if target in('system','gemm')else 0,
            selected_macro_lef if target in('system','gemm')else None,
            identity.get('sram_macro_name')))
        if stats.exists() and not args.map_from and not args.reuse_mapped:
            mesh=target in('array','system','gemm')
            arithmetic_units=rows*rows+(1 if target in('system','gemm')else 0)if mesh else 1
            expected=0 if stages==2 or target=='fp64' else 16*arithmetic_units
            if replicas!=expected:raise RuntimeError(f'local control copies merged or missing: {replicas} != {expected}')
            synthesis['expected_control_replica_instances']=expected
            pe_rtl=out/'inputs/SimdPE.sv'
            expected_contexts=16*(rows*rows if mesh else 1)if pe_rtl.exists()and'SimdContextReadReplicas'in pe_rtl.read_text()else 0
            if context_replicas!=expected_contexts:raise RuntimeError(f'local context read copies merged: {context_replicas} != {expected_contexts}')
            synthesis['expected_context_read_replica_instances']=expected_contexts
            expected_packed_a=16*rows*rows if mesh and pe_rtl.exists()and'SimdPackedAReplicas'in pe_rtl.read_text()else 0
            if packed_a_replicas!=expected_packed_a:raise RuntimeError(f'local packed A sources merged: {packed_a_replicas} != {expected_packed_a}')
            synthesis['expected_packed_a_register_instances']=expected_packed_a
            for name,actual in [('SimdObservationSelect',observation_replicas),('SimdStateWriteSelect',write_replicas)]:
                candidates=('SimdObservationSelect','SimdObservationState')if name=='SimdObservationSelect'else(name,)
                expected_local=16*(rows*rows if mesh else 1)if pe_rtl.exists()and any(n in pe_rtl.read_text()for n in candidates)else 0
                if actual!=expected_local:raise RuntimeError(f'{name} copies merged: {actual} != {expected_local}')
            expected_commit=16*(rows*rows if mesh else 1)if pe_rtl.exists()and'SimdPe'in pe_rtl.read_text()and'CommitWord'in pe_rtl.read_text()else 0
            if commit_replicas!=expected_commit:raise RuntimeError(f'local commit words merged: {commit_replicas} != {expected_commit}')
            expected_seed=16*(rows*rows if mesh else 1)if pe_rtl.exists()and'SimdPeSeedWord'in pe_rtl.read_text()else 0
            if seed_replicas!=expected_seed:raise RuntimeError(f'local seed words merged: {seed_replicas} != {expected_seed}')
            expected_valid=16 if target=='pe'and json.loads((out/'generated/capabilities.json').read_text()).get('fixed_state_commit')else 0
            if valid_replicas!=expected_valid:raise RuntimeError(f'local response validity copies merged: {valid_replicas} != {expected_valid}')
            capfile=out/'generated/capabilities.json'
            expected_preparation=16*arithmetic_units if capfile.exists()and json.loads(capfile.read_text()).get('registered_preparation_controls')else 0
            if preparation_replicas!=expected_preparation:raise RuntimeError(f'local preparation controls merged: {preparation_replicas} != {expected_preparation}')
            array_rtl=out/'inputs/SystolicArray.sv'
            if mesh and array_rtl.exists()and'SimdWaveRegister'in array_rtl.read_text():
                hop=1 if stages==2 else 2;cols=identity['cols']
                expected_waves=hop*(rows*(rows-1)//2+cols*(cols-1)//2+2*rows*cols)
                if wave_replicas!=expected_waves:raise RuntimeError(f'local wave banks merged: {wave_replicas} != {expected_waves}')
                synthesis['expected_wave_register_instances']=expected_waves
        # This inventory also applies when routing an authenticated existing
        # mapping: replicated state must exist in that netlist, not just RTL.
        capfile=out/'generated/capabilities.json'
        if capfile.exists()and json.loads(capfile.read_text()).get('distributed_request_capture'):
            if elastic_word_replicas!=16:
                raise RuntimeError(f'distributed request capture words merged or missing: {elastic_word_replicas} != 16')
            synthesis['expected_elastic_preparation_word_instances']=16
        if capfile.exists()and json.loads(capfile.read_text()).get('local_observation_initialization'):
            expected_local=16*(rows*rows if target in('array','system','gemm')else 1)
            if observation_state_replicas!=expected_local or observation_state_flops!=16:
                raise RuntimeError(f'local initialized state missing: {observation_state_replicas} instances, {observation_state_flops} FFs each; expected {expected_local} x 16')
            synthesis['expected_observation_state_instances']=expected_local
        (out/'synthesis.json').write_text(json.dumps(synthesis,indent=2)+'\n')
    if args.synth_only:print(str(out));return
    (out/'timing.json').unlink(missing_ok=True)
    sdc=out/'design.sdc';sdc.write_text(f'''set_cmd_units -time ns -capacitance fF
create_clock -name core -period {period:.6f} [get_ports clock]
set_clock_uncertainty 0.050 [get_clocks core]
set data_inputs [lsearch -all -inline -not -exact [all_inputs] [get_ports clock]]
set_input_delay 0.100 -clock core $data_inputs
set_input_transition 0.050 $data_inputs
set_output_delay 0.100 -clock core [all_outputs]
set_load 3.898 [all_outputs]
''')
    if target in ('system','gemm'):
        from mapped_system_constraints import write_system_sdc
        shutil.copy2(ROOT/'physical/cdc_constraints.tcl',out/'cdc_constraints.tcl')
        shutil.copy2(ROOT/'physical/mapped_system_constraints.py',out/'mapped_system_constraints.py')
        write_system_sdc(out/'synth.v',sdc,out/'cdc_constraints.tcl')
        identity['cdc_helper_sha256']=sha(out/'cdc_constraints.tcl')
        identity['mapped_constraint_generator_sha256']=sha(out/'mapped_system_constraints.py')
        identity_file.write_text(json.dumps(identity,indent=2)+'\n')
        identity['memory_macro_instances']=16
    # Large carry-save buses can produce assignment/port lines beyond the
    # OpenSTA Verilog lexer's buffer. Preserve the mapped netlist and import
    # an authenticated whitespace-only copy; do not change connectivity.
    from format_mapped_verilog import wrap_assignments
    tool_net=out/'tool_input.v'
    if args.resume_cts or args.resume_initial_hold or args.resume_global_route or args.resume_detailed_route:
        checkpoint=json.loads((out/'cts_timing.json').read_text())
        if checkpoint.get('tool_input_netlist_sha256'):
            for key,name in [('tool_input_netlist_sha256','tool_input.v'),
                    ('tool_input_formatter_sha256','netlist_formatter.py')]:
                if sha(out/name)!=checkpoint[key]:raise ValueError('CTS import snapshot changed: '+name)
                identity[key]=checkpoint[key]
            identity['tool_input_transform']=checkpoint['tool_input_transform']
            identity['tool_input_wrapped_lines']=checkpoint['tool_input_wrapped_lines']
        else:tool_net=out/'synth.v'
        identity['resumed_cts_report_sha256']=sha(out/'cts_timing.json')
    elif args.scalarize_tool_input:
        script=out/'scalarize_tool_input.ys'
        script.write_text(f'read_liberty -lib {lib}\nread_verilog {out/"synth.v"}\nhierarchy -check -top {identity["top"]}\nsplitnets\nwrite_verilog -noattr -noexpr {tool_net}\n')
        with(out/'scalarize_tool_input.log').open('w')as log:
            subprocess.run([tool('yosys'),'-Q','-T','-s',str(script)],env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        identity['tool_input_scalarization_script_sha256']=sha(script)
    else:shutil.copy2(out/'synth.v',tool_net)
    if not(args.resume_cts or args.resume_initial_hold or args.resume_global_route or args.resume_detailed_route):
        wrapped=wrap_assignments(tool_net,all_statements=True)
        shutil.copy2(ROOT/'physical/format_mapped_verilog.py',out/'netlist_formatter.py')
        identity.update(tool_input_netlist_sha256=sha(tool_net),
            tool_input_formatter_sha256=sha(out/'netlist_formatter.py'),
            tool_input_wrapped_lines=wrapped,
            tool_input_transform=('Yosys splitnets; 'if args.scalarize_tool_input else'')+'whitespace only; identical non-whitespace token sequence')
    identity_file.write_text(json.dumps(identity,indent=2)+'\n')
    env.update(SIMD_TECH_LEF=str(technology['tech_lef']),
      SIMD_CELL_LEF=str(technology['cell_lef']),SIMD_LIB=str(lib),
      SIMD_NETLIST=str(tool_net),SIMD_TOP=identity['top'],SIMD_SDC=str(sdc),
      SIMD_TRACKS=str(technology['tracks']),SIMD_RCX=str(technology['rcx']),SIMD_OUT=str(out))
    shutil.copy2(ROOT/'physical/protected_ties.tcl',out/'protected_ties.tcl')
    identity['protected_ties_sha256']=sha(out/'protected_ties.tcl')
    identity_file.write_text(json.dumps(identity,indent=2)+'\n')
    env['SIMD_PROTECTED_TIES']=str(out/'protected_ties.tcl')
    shutil.copy2(ROOT/'physical/protected_clocks.tcl',out/'protected_clocks.tcl')
    identity['protected_clocks_sha256']=sha(out/'protected_clocks.tcl')
    identity_file.write_text(json.dumps(identity,indent=2)+'\n')
    env['SIMD_PROTECTED_CLOCKS']=str(out/'protected_clocks.tcl')
    env.update(SIMD_CTS_ROOT=technology['cts_root'],SIMD_CTS_BUFFERS=' '.join(technology['cts_buffers']),SIMD_SET_RC=str(pdk/'setRC.tcl'))
    if target in ('system','gemm'):
        env['SIMD_SYSTEM']='1'
        env['SIMD_MACRO_LEF']=str(selected_macro_lef)
        env['SIMD_SRAM_MASTER']=identity['sram_macro_name']
        env['SIMD_MACRO_LIB']=str(selected_macro_lib)
        for source,name,key in [
                (ROOT/'physical/sram_floorplan.tcl','sram_floorplan.tcl','macro_floorplan_sha256'),
                (ROOT/'physical/protected_power.tcl','protected_power.tcl','protected_power_sha256'),
                (pdk/'openRoad/pdn/grid_strategy-M1-M2-M5-M6.tcl','pdn.tcl','power_grid_strategy_sha256')]:
            snapshot=out/name
            if snapshot.exists():
                if identity.get(key)!=sha(snapshot):raise ValueError('system physical snapshot changed: '+name)
            else:shutil.copy2(source,snapshot)
            identity[key]=sha(snapshot)
        identity_file.write_text(json.dumps(identity,indent=2)+'\n')
        env['SIMD_MACRO_FLOORPLAN']=str(out/'sram_floorplan.tcl')
        env['SIMD_PDN_SCRIPT']=str(out/'pdn.tcl')
    env['SIMD_HOLD_MARGIN']=str(identity['routed_hold_repair_margin_ns'])
    env['SIMD_SETUP_MARGIN']=str(identity['setup_repair_margin_ns'])
    if identity['placement_profile']=='geometric':env['SIMD_GEOMETRIC_PLACEMENT']='1'
    if identity.get('placement_grid_count'):env['SIMD_PLACE_GRID']=str(identity['placement_grid_count'])
    clock_options=identity.get('clock_tree_options')
    if clock_options:
        env.update(SIMD_CTS_ROOT=clock_options['root'],SIMD_CTS_BUFFERS=' '.join(clock_options['buffers']))
        if clock_options.get('sink_clustering'):
            env.update(SIMD_CTS_CLUSTER_SIZE=str(clock_options['cluster_size']),
              SIMD_CTS_DIAMETER=str(clock_options['cluster_diameter_um']))
    if identity.get('pe_hierarchy'):
        fence=out/'pe_fences.tcl'
        if not fence.exists():
            shutil.copy2(ROOT/'physical/pe_fences.tcl',fence)
            identity['fence_sha256']=sha(fence);identity_file.write_text(json.dumps(identity,indent=2)+'\n')
        elif identity.get('fence_sha256')!=sha(fence):raise RuntimeError('placement fence snapshot changed')
        env['SIMD_ARRAY_ROWS']=str(rows);env['SIMD_ARRAY_COLS']=str(identity['cols'])
        env['SIMD_FENCE_SCRIPT']=str(fence)
    if identity['placement_profile']=='regional':
        if not identity.get('pe_hierarchy')or rows<=2:
            raise ValueError('regional initial placement requires a fenced array larger than 2x2')
        seed=out/'seed_regions.tcl'
        if not seed.exists():shutil.copy2(ROOT/'physical/seed_regions.tcl',seed)
        if identity.get('regional_seed_sha256')and identity['regional_seed_sha256']!=sha(seed):
            raise RuntimeError('regional placement seed script changed')
        identity['regional_seed_sha256']=sha(seed);identity_file.write_text(json.dumps(identity,indent=2)+'\n')
        env['SIMD_REGION_SEED_SCRIPT']=str(seed)
    if args.cts_only:env['SIMD_CTS_ONLY']='1'
    if args.detailed_route_only:env['SIMD_DETAILED_ROUTE_ONLY']='1'
    if args.resume_cts:
        checkpoint=json.loads((out/'cts_timing.json').read_text())
        if checkpoint['mapped_sha256']!=identity['mapped_sha256']or checkpoint['checkpoint_sha256']!=sha(out/'post_cts.odb'):
            raise RuntimeError('CTS checkpoint is from a different mapped implementation')
        env['SIMD_RESUME_CTS']='1'
    if args.resume_initial_hold or args.resume_global_route or args.resume_detailed_route:
        checkpoint_path=out/'resume_checkpoint_manifest.json'
        checkpoint=json.loads(checkpoint_path.read_text())
        resume_stage='post_detailed_route'if args.resume_detailed_route else'post_global_route'if args.resume_global_route else'post_initial_hold'
        if checkpoint['stage']!=resume_stage or checkpoint['rtl_sha256']!=identity['rtl_sha256'] or checkpoint['mapped_sha256']!=identity['mapped_sha256']:
            raise ValueError('post-hold checkpoint implementation differs')
        required=[('checkpoint_sha256',resume_stage+'.odb'),('sdc_sha256','design.sdc')]
        if args.resume_global_route:required.append(('guides_sha256','post_global_route.guide'))
        if args.resume_detailed_route:required.append(('routed_netlist_sha256','post_detailed_route.v'))
        for key,name in required:
            if sha(out/name)!=checkpoint[key]:raise ValueError('post-hold checkpoint changed: '+name)
        parent=Path(checkpoint['parent_manifest'])
        if sha(parent)!=checkpoint['parent_manifest_sha256']:raise ValueError('post-hold parent evidence changed')
        for path,digest in json.loads(parent.read_text())['evidence'].items():
            if sha(Path(path))!=digest:raise ValueError('post-hold parent input changed: '+path)
        identity['resumed_stage']=resume_stage
        identity['resumed_stage_manifest_sha256']=sha(checkpoint_path)
        identity_file.write_text(json.dumps(identity,indent=2)+'\n')
        env['SIMD_RESUME_DETAILED_ROUTE'if args.resume_detailed_route else'SIMD_RESUME_GLOBAL_ROUTE'if args.resume_global_route else'SIMD_RESUME_INITIAL_HOLD']='1'
    flow=out/'flow.tcl'
    if (out/'route.log').exists():
        attempt=len(list(out.glob('attempt_*_route.log')))+1
        for name in('route.log','flow.tcl','failure.json'):
            if(out/name).exists():shutil.copy2(out/name,out/f'attempt_{attempt:03d}_{name}')
    shutil.copy2(ROOT/'physical/run.tcl',flow)
    try:
        with(out/'route.log').open('w')as log:
            subprocess.run([tool('openroad'),'-exit',str(flow)],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        if args.detailed_route_only:
            log_text=(out/'route.log').read_text()
            if 'SIMD_DETAILED_ROUTE_CHECKPOINT_WRITTEN'not in log_text:
                raise RuntimeError('detailed route did not reach its atomic geometry checkpoint')
            paths=[out/name for name in('post_detailed_route.odb','post_detailed_route.v',
                'design.sdc','synth.v','build_identity.json','route.log','flow.tcl','physical_driver.py')]
            paths += [out/'route_parent.json']if(out/'route_parent.json').exists()else[]
            checkpoint=dict(stage='post_detailed_route',rtl_sha256=identity['rtl_sha256'],
                mapped_sha256=identity['mapped_sha256'],arithmetic_stages=2,core_period_ns=period,
                extracted_sta_complete=False,physical_frequency_certified=False,
                evidence={str(path):sha(path)for path in paths})
            (out/'detailed_route_checkpoint.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
            (out/'failure.json').unlink(missing_ok=True);print(json.dumps(checkpoint,indent=2));return
        if args.cts_only:
            text=(out/'route.log').read_text()
            def cts_slack(marker):return float(re.search(marker+r'.*?worst slack\s+([-+\d.]+)',text,re.S)[1])
            report={**identity,'timing_status':'cts_only_estimate','period_ns':period,
              'setup_wns_ns':cts_slack('SIMD_CTS_SETUP'),'hold_wns_ns':cts_slack('SIMD_CTS_HOLD'),
              'checkpoint_sha256':sha(out/'post_cts.odb'),'post_route_certified':False}
            (out/'cts_timing.json').write_text(json.dumps(report,indent=2)+'\n')
            (out/'failure.json').unlink(missing_ok=True);print(json.dumps(report,indent=2));return
        subprocess.run([sys.executable,str(ROOT/'physical/finalize.py'),str(out)],check=True,stdout=subprocess.DEVNULL,env=env)
        report=json.loads((out/'timing.json').read_text())
        (out/'failure.json').unlink(missing_ok=True);print(json.dumps(report,indent=2))
    except Exception as e:
        failure={'status':'flow_failed','error':str(e),'build_identity':str(identity_file),'log':str(out/'route.log')}
        (out/'failure.json').write_text(json.dumps(failure,indent=2)+'\n');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corner',choices=['TT','SS','FF'],default='TT')
    p.add_argument('--scalarize-tool-input',action='store_true',help='split internal buses for the OpenSTA importer, retaining the original mapped netlist')
    p.add_argument('--allow-unverified-sram',action='store_true',help='exploratory system P&R only; cannot produce a physical acceptance certificate')
    p.add_argument('--sram-package',type=Path,help='authenticated generated research macro package; estimated timing cannot certify frequency')
    p.add_argument('--target',choices=['fp64','int64','alu','pe','array','system','gemm'],default='fp64')
    p.add_argument('--stages',type=int,choices=[2],default=2);p.add_argument('--rows',type=int,default=1)
    p.add_argument('--period',type=float,default=1);p.add_argument('--label');p.add_argument('--reuse-mapped',action='store_true')
    p.add_argument('--pe-hierarchy',action='store_true')
    p.add_argument('--rtl-build',type=Path,help='synthesize the authenticated frozen RTL of an existing simulator without re-elaboration')
    p.add_argument('--mapping-profile',choices=['full','fast','choices'],help='ABC mapping search policy; choices bounds its isolated combinational search to600s')
    p.add_argument('--sequential-profile',choices=['standard'],help='ASAP7 standard sequential mapping')
    p.add_argument('--clock-profile',choices=['standard','clustered','clustered64'],help='record actual clock buffer and sink-clustering policy')
    p.add_argument('--hold-margin',type=float,help='routed hold-repair guard band in ns; acceptance uses strict extracted STA')
    p.add_argument('--setup-margin',type=float,help='setup-repair guard band in ns; does not change the SDC or acceptance period')
    p.add_argument('--placement-profile',choices=['timing','geometric','regional'],help='global placement optimization; final extracted STA is unchanged')
    p.add_argument('--placement-grid',type=int,help='bounded GPL grid, power of two from 32 to 2048; detailed placement and timing checks stay unchanged')
    p.add_argument('--map-from',type=Path);p.add_argument('--synth-only',action='store_true');p.add_argument('--cts-only',action='store_true');p.add_argument('--resume-cts',action='store_true');p.add_argument('--resume-initial-hold',action='store_true');p.add_argument('--resume-global-route',action='store_true');p.add_argument('--resume-detailed-route',action='store_true');p.add_argument('--detailed-route-only',action='store_true');p.add_argument('--remap-premap',action='store_true');p.add_argument('--free-running',action='store_true');run(p.parse_args())
