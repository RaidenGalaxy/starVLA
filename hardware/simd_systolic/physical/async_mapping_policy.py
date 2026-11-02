"""Bind asynchronous ASAP7 mapping to actual Liberty event simulation evidence."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

POLICY='explicit_asap7_async_logical_q_v1'
MASTER='DFFASRHQNx1_ASAP7_75t_R'


def count_async(path):
    count=0
    with Path(path).open()as stream:
        for line in stream:
            match=re.match(r'\s*(DFFAS\w+)\s',line)
            if match:
                if match[1]!=MASTER:raise ValueError('unsupported asynchronous cell: '+match[1])
                count+=1
    return count


def qualify(out,library):
    from run import ROOT,sha
    mapping=ROOT/'physical/Asap7AsyncFlops.v'
    proof=out/'async_mapping_proof'
    suffix=0
    while proof.exists():
        suffix+=1;proof=out/f'async_mapping_proof{suffix}'
    with(out/'async_mapping_qualification.log').open('w')as stream:
        subprocess.run([sys.executable,str(ROOT/'physical/check_async_mapping.py'),
            '--out',str(proof),'--liberty',str(library)],check=True,
            stdout=stream,stderr=subprocess.STDOUT,timeout=300)
    shutil.copy2(mapping,out/mapping.name)
    result=json.loads((proof/'verification.json').read_text())
    if result['mapping_sha256']!=sha(out/mapping.name):raise ValueError('mapping changed during qualification')
    return dict(async_mapping_policy=POLICY,async_mapping_sha256=sha(out/mapping.name),
        async_mapping_proof_file=str(proof/'verification.json'),
        async_mapping_proof_sha256=sha(proof/'verification.json'))


def validate(out,identity,netlist=None):
    from run import sha
    count=count_async(netlist or out/'synth.v')
    if not count:return dict(async_cells=0,qualified=True)
    if identity.get('async_mapping_policy')!=POLICY:
        raise ValueError('unqualified asynchronous ASAP7 mapping; remap or apply a verified reset-pin correction')
    mapping=out/'Asap7AsyncFlops.v'
    if sha(mapping)!=identity['async_mapping_sha256']:raise ValueError('asynchronous mapping snapshot changed')
    path=Path(identity['async_mapping_proof_file'])
    if sha(path)!=identity['async_mapping_proof_sha256']:raise ValueError('asynchronous mapping proof changed')
    proof=json.loads(path.read_text())
    if not proof.get('passed')or proof['liberty_sha256']!=identity['liberty_sha256']or proof['mapping_sha256']!=sha(mapping):
        raise ValueError('asynchronous mapping proof does not match the actual library and mapping')
    for filename,digest in proof['evidence'].items():
        if sha(filename)!=digest:raise ValueError('asynchronous mapping evidence changed: '+filename)
    return dict(async_cells=count,qualified=True,policy=POLICY,proof_sha256=sha(path))


def validate_checkpoint(source,prior):
    """Require a corrected physical lineage before any further system ECO."""
    from run import sha
    if prior.get('qualified_async_mapping_root'):
        root=Path(prior['qualified_async_mapping_root'])
        identity=json.loads((root/'build_identity.json').read_text())
        if identity['mapped_sha256']!=prior['source_mapped_sha256']or sha(root/'synth.v')!=identity['mapped_sha256']:
            raise ValueError('qualified system mapping source changed')
        return validate(root,identity)
    if not prior.get('async_mapping_correction_verified'):
        raise ValueError('legacy system asynchronous mapping is unqualified; correct all reset pins first')
    proof=Path(prior['async_correction_verification'])
    if sha(proof)!=prior['async_correction_verification_sha256']:
        raise ValueError('system asynchronous correction proof changed')
    record=json.loads(proof.read_text())
    if not record.get('passed')or not record.get('all_non_async_connectivity_and_placement_preserved'):
        raise ValueError('system asynchronous correction was not completed')
    for filename,digest in record['evidence'].items():
        if sha(filename)!=digest:raise ValueError('system asynchronous correction evidence changed: '+filename)
    identity={**record['actual_liberty_event_proof'],'liberty_sha256':record['liberty_sha256']}
    return validate(proof.parent,identity,proof.parent/'repaired.v')
