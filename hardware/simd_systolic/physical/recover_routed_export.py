#!/usr/bin/env python3
"""Recover a routed F2 I64 export without changing placement or routing."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from run import ROOT, sha, tool, summarize
from asap7_platform import profile


def run(args):
    source=args.source.resolve();out=args.out.resolve()
    proof=source/'detailed_route_checkpoint.json';record=json.loads(proof.read_text())
    identity=json.loads((source/'build_identity.json').read_text())
    if (identity['scope'],identity['stages'],record['core_period_ns'])!=('int64',2,1):
        raise ValueError('requires an authenticated routed F2 I64 component at1ns')
    for path,digest in record['evidence'].items():
        if sha(Path(path))!=digest:raise ValueError('routed parent evidence changed: '+path)
    database=source/'post_detailed_route.odb'
    if record['evidence'].get(str(database))!=sha(database):raise ValueError('parent geometry is not bound')
    technology=profile('TT')
    if sha(technology['liberty'])!=identity['liberty_sha256']:raise ValueError('process library changed')
    if sha(source/'synth.v')!=identity['mapped_sha256']:raise ValueError('original mapping changed')
    if out.exists():raise ValueError('preserve old runs; choose another output')
    out.mkdir(parents=True)
    for name in('build_identity.json','synth.v','design.sdc','physical_driver.py','cts_timing.json','post_cts.odb'):
        shutil.copy2(source/name,out/name)
    shutil.copytree(source/'inputs',out/'inputs')
    shutil.copy2(__file__,out/'recovery_driver.py')
    shutil.copy2(ROOT/'physical/canonical_net_names.tcl',out/'canonical_net_names.tcl')
    drc=source/'route.drc'
    if not drc.exists()or drc.read_text().strip():raise ValueError('source has unresolved or missing detailed-route DRC')
    shutil.copy2(drc,out/'route.drc')
    state=dict(kind='canonical_routed_export_recovery',passed=False,arithmetic_stages=2,
        core_period_ns=1,physical_frequency_certified=False,steps=[],source_checkpoint=str(proof),
        source_checkpoint_sha256=sha(proof),source_database_sha256=sha(database))
    def save():
        pending=out/'recovery_state.json.pending'
        pending.write_text(json.dumps(state,indent=2)+'\n');pending.replace(out/'recovery_state.json')
    local=ROOT/'.deps/tools/openroad';env={**os.environ,'LD_LIBRARY_PATH':
        str(local/'opt/or-tools/lib')+':'+str(local/'usr/lib/x86_64-linux-gnu'),
        'SIMD_OUT':str(out),'SIMD_RCX':str(technology['rcx'])}
    def call(name,command,timeout):
        log=out/(name+'.log');state['active_phase']=name;save()
        with log.open('w')as stream:
            process=subprocess.Popen(command,env=env,cwd=ROOT,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
            state['active_pid']=process.pid;save();timed_out=False
            try:code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out=True
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:code=process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);code=process.wait()
        state['steps'].append(dict(name=name,command=command,returncode=code,timed_out=timed_out,
            log=str(log),log_sha256=sha(log)));state.pop('active_pid',None);save()
        if code or timed_out:raise RuntimeError('recovery phase failed: '+name)
    try:
        script=out/'canonicalize.tcl';script.write_text(f'''set_thread_count 8
read_liberty {{{technology['liberty']}}}
read_db {{{database}}}
source {{{out/'canonical_net_names.tcl'}}}
simd_export_canonical_net_names {{{out}}}
''')
        call('canonicalize',[tool('openroad'),'-exit',str(script)],600)
        if sha(out/'before_rename.def')!=sha(out/'restored_names.def'):
            raise ValueError('canonical export changed full physical geometry/connectivity on roundtrip')
        if sha(out/'before_connectivity.tcllist')!=sha(out/'canonical_connectivity.tcllist'):
            raise ValueError('canonical export changed a cell, connection, wire object, port or protection')
        files=[proof,database,drc,*[out/n for n in('canonical.odb','routed.v','net_names.tsv',
            'before_rename.def','restored_names.def','before_connectivity.tcllist',
            'canonical_connectivity.tcllist','canonicalize.log','canonicalize.tcl',
            'canonical_net_names.tcl','recovery_driver.py','design.sdc')]]
        export=dict(geometry_and_connectivity_preserved=True,top_ports_preserved=True,
            cell_masters_and_protection_preserved=True,only_internal_net_labels_changed=True,
            physical_frequency_certified=False,evidence={str(p):sha(p)for p in files})
        manifest=out/'canonical_export.json';manifest.write_text(json.dumps(export,indent=2)+'\n')
        identity.update(canonical_export_manifest_sha256=sha(manifest),
            export_recovery_driver_sha256=sha(out/'recovery_driver.py'))
        (out/'build_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        state['canonical_geometry_and_connectivity_preserved']=True;save()
        tail=(ROOT/'physical/run.tcl').read_text().split('define_process_corner -ext_model_index 0 X\n',1)[1]
        script=out/'extract.tcl';script.write_text(f'''set_thread_count 8
read_liberty {{{technology['liberty']}}}
read_db {{{out/'canonical.odb'}}}
set_cmd_units -time ns -capacitance fF
read_sdc {{{out/'design.sdc'}}}
set_propagated_clock [all_clocks]
source {{{technology['root']/'setRC.tcl'}}}
set_wire_rc -signal -layer M3
set_wire_rc -clock -layer M5
define_process_corner -ext_model_index 0 X
'''+tail)
        shutil.copy2(script,out/'flow.tcl')
        call('route',[tool('openroad'),'-exit',str(script)],args.timeout)
        timing=summarize(out,identity,1)
        if not timing.get('parasitic_annotation_clean'):
            raise ValueError('routed STA has no complete, error-free parasitic annotation')
        state.update(extracted_sta_completed=True,timing_sha256=sha(out/'timing.json'),
            setup_wns_ns=timing['setup_wns_ns'],hold_wns_ns=timing['hold_wns_ns'])
        call('routed_numerical',[sys.executable,str(ROOT/'physical/check_int64.py'),
            str(out),'--routed','--count','1000'],1800)
        numerical=out/'check_routed/verification.json';checked=json.loads(numerical.read_text())
        if not checked['passed']or checked['netlist_sha256']!=sha(out/'routed.v'):
            raise ValueError('routed numerical evidence does not match recovered netlist')
        if sha(database)!=state['source_database_sha256']:raise ValueError('source geometry changed')
        state.update(passed=True,routed_numerical_sha256=sha(numerical))
    except Exception as error:
        state['error']=str(error);raise
    finally:
        state.pop('active_phase',None);save()
        (out/'recovery_verification.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--timeout',type=int,default=7200)
    args=parser.parse_args()
    if not 1<=args.timeout<=7200:parser.error('extraction timeout must be within1..7200seconds')
    run(args)
