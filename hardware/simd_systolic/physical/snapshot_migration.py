#!/usr/bin/env python3
"""Archive current migration sources and a diff against the authenticated M0.

Historical archives are immutable. The current manifest is an index, not a
claim that experimental or physically unqualified RTL has passed acceptance.
"""
import argparse
import difflib
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOLDERS=('configs','physical','python','rtl','scripts','sim','src','tests','vendor','project')
TOP=('Makefile','build.sbt','dependencies.lock.json','README.md','MIGRATION_ASAP7.md',
     'STATUS.md','VERIFICATION.md','VERIFICATION_RESULTS.md')


def sha(data):return hashlib.sha256(data).hexdigest()


def run(out):
    baseline=out/'baseline';manifest=baseline/'manifest.json'
    frozen=json.loads(manifest.read_text());original=baseline/'source.tar.gz'
    if sha(original.read_bytes())!=frozen['archive_sha256']:
        raise ValueError('baseline archive changed')
    before={}
    with tarfile.open(original,'r:gz')as archive:
        for name,digest in frozen['source_sha256'].items():
            content=archive.extractfile(name).read()
            if sha(content)!=digest:raise ValueError('baseline member changed: '+name)
            before[name]=content
    paths={ROOT/name for name in TOP if(ROOT/name).is_file()}
    for folder in FOLDERS:
        paths.update(path for path in(ROOT/folder).rglob('*')if path.is_file()and
            not any(part in ('__pycache__','target','.bloop','.metals')for part in path.relative_to(ROOT).parts)
            and path.suffix!='.pyc')
    after={str(path.relative_to(ROOT)):path.read_bytes()for path in sorted(paths)}
    hashes={name:sha(content)for name,content in after.items()}
    tree=sha(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode())
    changes=[];patch=[]
    for name in sorted(before.keys()|after.keys()):
        old=before.get(name);new=after.get(name)
        if old==new:continue
        changes.append(dict(path=name,status='added'if old is None else'deleted'if new is None else'modified',
            before_sha256=sha(old)if old is not None else None,after_sha256=sha(new)if new is not None else None))
        # All archived source files must be textual for a reviewable patch.
        left=old.decode().splitlines(keepends=True)if old is not None else[]
        right=new.decode().splitlines(keepends=True)if new is not None else[]
        for line in difflib.unified_diff(left,right,fromfile='a/'+name if old is not None else'/dev/null',
                tofile='b/'+name if new is not None else'/dev/null'):
            patch.append(line if line.endswith('\n')else line+'\n\\ No newline at end of file\n')
    destination=out/'implementation';destination.mkdir(exist_ok=True)
    name=destination/('source_'+tree[:12]+'.tar.gz')
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w',format=tarfile.PAX_FORMAT)as archive:
        for relative,content in after.items():
            member=tarfile.TarInfo(relative);member.size=len(content)
            member.mode=0o755 if (ROOT/relative).stat().st_mode&0o111 else 0o644
            archive.addfile(member,io.BytesIO(content))
    compressed=gzip.compress(buffer.getvalue(),mtime=0)
    if name.exists()and name.read_bytes()!=compressed:
        raise ValueError('an archive with this content identity already differs')
    # Recheck after reading so the manifest cannot describe a mixed source edit.
    if any((ROOT/relative).read_bytes()!=content for relative,content in after.items()):
        raise ValueError('source changed during snapshot')
    name.write_bytes(compressed)
    patch_path=out/'changes.patch';patch_path.write_text(''.join(patch))
    (out/'change_inventory.json').write_text(json.dumps(dict(files=changes,total_changed=len(changes)),indent=2)+'\n')
    result=dict(kind='migration_source_snapshot',source_sha256=hashes,tree_sha256=tree,
        archive=str(name.resolve()),archive_sha256=sha(compressed),
        baseline_manifest_sha256=sha(manifest.read_bytes()),changes_patch_sha256=sha(patch_path.read_bytes()),
        experiment_scope='Drivers are archived; separately identified build snapshots keep their own inputs and evidence.')
    (destination/'source_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(archive=str(name),source_files=len(hashes),changed_files=len(changes),tree_sha256=tree)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/migration_asap7_f2')
    run(p.parse_args().out.resolve())
