#!/usr/bin/env python3
"""Resume authenticated placement or routing in a separate iteration."""
import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from run import ROOT,sha,tool,summarize,ARCH


def reclock(args):
    source=args.source.resolve();identity=json.loads((source/'build_identity.json').read_text())
    if identity.get('stages')!=2 or identity.get('process')!='asap7' or identity.get('architecture_sha256')!=ARCH['architecture_sha256']:
        raise ValueError('new physical iterations require the current ASAP7 F2 architecture')
    if identity['scope']in('system','gemm'):raise ValueError('use the main multiclock physical entry for SRAM systems')
    if identity.get('analysis_period_ns',identity['mapping_period_ns'])!=1:raise ValueError('core acceptance remains fixed at 1ns')
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,47}',args.label):raise ValueError('invalid iteration label')
    if args.cluster_size<1 or args.cluster_diameter<=0:raise ValueError('invalid sink clustering')
    density_penalty=getattr(args,'initial_density_penalty',None)
    min_phi=getattr(args,'min_phi_coef',None)
    no_inflation=getattr(args,'no_routability_inflation',False)
    if (density_penalty is not None or min_phi is not None or no_inflation)and args.checkpoint!='seeded':
        raise ValueError('density search overrides require the seeded checkpoint')
    if density_penalty is not None and not 0<density_penalty<=1:
        raise ValueError('initial density penalty must be in (0,1]')
    if min_phi is not None and not .95<=min_phi<=1.05:
        raise ValueError('minimum density penalty update factor must be in [0.95,1.05]')
    if sha(source/'synth.v')!=identity['mapped_sha256']:raise ValueError('source mapped netlist changed')
    for name,digest in identity['input_manifest'].items():
        if sha(source/'inputs'/name)!=digest:raise ValueError('source RTL snapshot changed: '+name)
    lib=Path(identity['liberty_file'])
    if sha(lib)!=identity['liberty_sha256']:raise ValueError('source process library differs')
    library=lib.read_text()
    for cell in [args.root,*args.buffers]:
        if not re.fullmatch(r'\w+',cell)or not re.search(r'cell\s*\(\s*"?'+re.escape(cell)+r'"?\s*\)',library):
            raise ValueError('unknown clock buffer '+cell)
    out=source.parent/(source.name+'_'+args.label);out.mkdir(exist_ok=True)
    guard=(out/'.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if(out/'flow.tcl').exists():raise ValueError('iteration exists; choose another label')
    for name in('inputs','generated'):
        if(source/name).exists():shutil.copytree(source/name,out/name,dirs_exist_ok=True)
    for name in('synth.v','mapping.lib','cell_statistics.json','synth.log','synthesis.json','design.sdc',
                'NangateSyncControls.v','sequential_mapping_proof.json'):
        if(source/name).exists():shutil.copy2(source/name,out/name)
    # Historical runs incorrectly waived the synchronous core reset port.
    # Preserve that optimization input, then time reset in the new iteration.
    sdc=out/'design.sdc';original=sdc.read_text()
    corrected='\n'.join(line for line in original.splitlines()
        if line.strip()!='if {[llength [get_ports -quiet reset]]} {set_false_path -from [get_ports reset]}')+'\n'
    if re.search(r'set_false_path[^\n]*\breset\b',corrected):raise ValueError('unrecognized reset exception; review this SDC explicitly')
    if corrected!=original:
        (out/'seed_constraints.sdc').write_text(original);sdc.write_text(corrected)
        identity['constraint_change']='removed historical synchronous reset timing exception'
    checkpoint={'cts':'post_cts.odb','global':'post_global.odb','seeded':'seeded_placement.odb',
        'global_route':'post_global_route.odb'}[args.checkpoint]
    seed={'cts':'seed_cts.odb','global':'seed_global.odb','seeded':'seed_placement.odb',
        'global_route':'seed_global_route.odb'}[args.checkpoint]
    if args.checkpoint=='global_route' and args.cts_only:
        raise ValueError('a routing checkpoint cannot produce a CTS-only iteration')
    shutil.copy2(source/checkpoint,out/seed)
    if sha(source/checkpoint)!=sha(out/seed):raise ValueError('placement changed during copying')
    margin=args.hold_margin if args.hold_margin is not None else(.08 if identity['scope']=='fp64'else .05)
    if not 0<=margin<=.5:raise ValueError('hold repair margin must be in [0,0.5] ns')
    setup=args.setup_margin if args.setup_margin is not None else identity.get('setup_repair_margin_ns',.05)
    if not 0<=setup<=.5:raise ValueError('setup repair margin must be in [0,0.5] ns')
    identity.update(placement_checkpoint_from=str(source),checkpoint_stage=args.checkpoint,
        seed_checkpoint_sha256=sha(out/seed),routed_hold_repair_margin_ns=margin,setup_repair_margin_ns=setup,
        physical_worker_threads=args.threads,
        clock_tree_action='retained'if args.checkpoint in('cts','global_route')else'rebuilt')
    if args.checkpoint=='global_route':
        # route.guide may predate incremental timing repair. Its routes can
        # omit new buffers/nets. The post-global-route ODB contains the updated
        # dbGuide objects and is the authoritative routing checkpoint.
        identity.update(seed_routing_guides_source='checkpoint_database_dbGuide_objects',
            routing_action='resume_detailed_route_without_repeating_timing_repair')
        for name in('post_global_route_setup.rpt','post_global_route_hold.rpt'):
            if(source/name).exists():shutil.copy2(source/name,out/name)
    if args.checkpoint not in('cts','global_route'):
        identity.update(global_placement_from=str(source),seed_global_sha256=sha(out/seed),
            clock_tree_options=dict(root=args.root,buffers=args.buffers,sink_clustering=True,
                cluster_size=args.cluster_size,cluster_diameter_um=args.cluster_diameter))
    if args.checkpoint=='seeded':
        if args.placement_grid<32 or args.placement_grid>2048 or args.placement_grid&(args.placement_grid-1):
            raise ValueError('placement grid must be a power of two in [32,2048]')
        identity.update(placement_profile='regional_refinement',placement_grid_count=args.placement_grid,
            placement_target_overflow=.01,placement_routability_inflation=not no_inflation)
        if no_inflation:
            identity.pop('routability_check_overflow',None)
            identity['routing_feedback_stage']='global_route_after_legalization_and_cts'
        else:identity['routability_check_overflow']=.02
        if density_penalty is not None:identity['placement_initial_density_penalty']=density_penalty
        if min_phi is not None:identity['placement_min_phi_coef']=min_phi
    (out/'build_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    base=(ROOT/'physical/run.tcl').read_text()
    prelude=base[:base.index('if {[info exists ::env(SIMD_RESUME_CTS)]}')]
    tail=base[base.index('set_propagated_clock [all_clocks]'):]
    preparation='''read_db $::env(SIMD_OUT)/'''+seed+'''
read_sdc $::env(SIMD_OUT)/design.sdc
source $::env(SIMD_SET_RC)
set_wire_rc -signal -layer M3
set_wire_rc -clock -layer M5
simd_routing_layers
'''
    rebuild='''
detailed_placement
estimate_parasitics -placement
repair_design -cap_margin 20 -slew_margin 20
detailed_placement
clock_tree_synthesis -root_buf $::env(SIMD_CTS_ROOT) -buf_list $::env(SIMD_CTS_BUFFERS) \
 -sink_clustering_enable -sink_clustering_size $::env(SIMD_CTS_CLUSTER_SIZE) \
 -sink_clustering_max_diameter $::env(SIMD_CTS_DIAMETER) -balance_levels
'''
    density_options=''
    if density_penalty is not None:density_options+=f' -init_density_penalty {density_penalty:g}'
    if min_phi is not None:density_options+=f' -min_phi_coef {min_phi:g}'
    # RUDY inflation can diverge when restoring fenced, nearly converged
    # regions. This optional retry omits that placement heuristic; the normal
    # strict global/detailed routing, DRC, electrical checks and STA still run.
    routability_options=''if no_inflation else' -routability_driven -routability_check_overflow 0.02'
    refinement=(f'''global_placement -density 0.35 -skip_initial_place \\
 -bin_grid_count {args.placement_grid} -overflow 0.01{routability_options}{density_options}
write_db $::env(SIMD_OUT)/post_global.odb
'''if args.checkpoint=='seeded'else'')
    if args.checkpoint=='global_route':
        # Preserve the database's final dbGuide objects. Reading an earlier
        # external guide file here can replace routes repaired incrementally.
        flow=prelude+preparation+'''set_propagated_clock [all_clocks]
set guided_nets 0
foreach net [[ord::get_db_block] getNets] {
 if {[llength [$net getGuides]] > 0} {incr guided_nets}
}
if {$guided_nets == 0} {error "routing checkpoint has no database guides"}
puts "SIMD_CHECKPOINT_DATABASE_GUIDED_NETS $guided_nets"
'''+base[base.index('detailed_route -output_drc'):]
    else:
        flow=prelude+preparation+refinement+(''if args.checkpoint=='cts'else rebuild)+tail
    flow=flow.replace('set_thread_count 8',f'set_thread_count {args.threads}',1)
    (out/'flow.tcl').write_text(flow)
    env=os.environ.copy();local=ROOT/'.deps/tools/openroad';pdk=ROOT/'.deps/asap7'
    env['LD_LIBRARY_PATH']=str(local/'opt/or-tools/lib')+':'+str(local/'usr/lib/x86_64-linux-gnu')
    env.update(SIMD_OUT=str(out),SIMD_TECH_LEF=str(pdk/'lef/asap7_tech_1x_201209.lef'),
        SIMD_CELL_LEF=str(pdk/'lef/asap7sc7p5t_28_R_1x_220121a.lef'),SIMD_LIB=str(lib),
        SIMD_RCX=str(pdk/'rcx_patterns.rules'),SIMD_CTS_ROOT=args.root,SIMD_CTS_BUFFERS=' '.join(args.buffers),
        SIMD_SET_RC=str(pdk/'setRC.tcl'),
        SIMD_CTS_CLUSTER_SIZE=str(args.cluster_size),SIMD_CTS_DIAMETER=str(args.cluster_diameter))
    env['SIMD_HOLD_MARGIN']=str(margin)
    env['SIMD_SETUP_MARGIN']=str(setup)
    if args.cts_only:env['SIMD_CTS_ONLY']='1'
    try:
        with(out/'route.log').open('w')as log:
            subprocess.run([tool('openroad'),'-exit',str(out/'flow.tcl')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        if args.cts_only:
            text=(out/'route.log').read_text()
            def slack(marker):return float(re.search(marker+r'.*?worst slack\s+([-+\d.]+)',text,re.S)[1])
            report={**identity,'timing_status':'cts_only_estimate','period_ns':identity['analysis_period_ns'],
                'setup_wns_ns':slack('SIMD_CTS_SETUP'),'hold_wns_ns':slack('SIMD_CTS_HOLD'),
                'checkpoint_sha256':sha(out/'post_cts.odb'),'post_route_certified':False}
            (out/'cts_timing.json').write_text(json.dumps(report,indent=2)+'\n')
        else:report=summarize(out,identity,identity['analysis_period_ns'])
        print(json.dumps(report,indent=2))
    except Exception as error:
        log_path=out/'route.log'
        errors=re.findall(r'^\[ERROR[^\n]*',log_path.read_text(errors='replace'),re.M)if log_path.exists()else[]
        (out/'failure.json').write_text(json.dumps(dict(status='physical_iteration_failed',
            error=str(error),last_tool_error=errors[-1]if errors else None),indent=2)+'\n')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('--label',required=True)
    parser.add_argument('--root',default='BUFx8_ASAP7_75t_R');parser.add_argument('--buffers',nargs='+',default=['BUFx2_ASAP7_75t_R','BUFx4_ASAP7_75t_R','BUFx8_ASAP7_75t_R'])
    parser.add_argument('--cluster-size',type=int,default=20);parser.add_argument('--cluster-diameter',type=float,default=50)
    parser.add_argument('--checkpoint',choices=['global','cts','seeded','global_route'],default='global',help='retain a clock tree, refine a placement seed, or resume detailed routing')
    parser.add_argument('--threads',type=int,choices=[1,2,4,8],default=4)
    parser.add_argument('--placement-grid',type=int,default=512,help='bounded GPL grid for seeded refinement; all timing checks stay unchanged')
    parser.add_argument('--initial-density-penalty',type=float,help='seeded placement density-force search parameter; does not change timing constraints')
    parser.add_argument('--min-phi-coef',type=float,help='lower bound of the seeded placement density-penalty update factor')
    parser.add_argument('--no-routability-inflation',action='store_true',help='seeded retry without RUDY area inflation; all routing, electrical and extracted timing checks remain enabled')
    parser.add_argument('--hold-margin',type=float,help='routed hold-repair guard band in ns; acceptance still uses strict extracted STA')
    parser.add_argument('--setup-margin',type=float,help='setup optimization guard band in ns; final timing constraints stay unchanged')
    parser.add_argument('--cts-only',action='store_true');reclock(parser.parse_args())
