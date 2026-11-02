#!/usr/bin/env python3
"""Fetch fixed Trilinos source directories in small Git transfer batches."""
import json
import os
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'.deps/xyce-native'
PACKAGES=('amesos','amesos2','aztecoo','belos','common','epetra','epetraext','ifpack',
          'kokkos','kokkoskernels','nox','rol','sacado','shylu','stokhos','teuchos',
          'tpetra','trilinoscouplings','triutils','zoltan','zoltan2','rtop','thyra','galeri')
DENY={'test','tests','example','examples','doc','docs','data','perf_test','benchmark','benchmarks'}


def populate(path,item):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'GIT_TERMINAL_PROMPT':'0'}
    log_path=BASE/'trilinos_source_batches.log'
    with log_path.open('w')as log:
        def git(*args):
            subprocess.run(['git','-c','http.version=HTTP/1.1',*args],cwd=path,env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        def read(*args):return subprocess.check_output(['git',*args],cwd=path,text=True)
        if not(path/'.git').exists():
            git('init');git('remote','add','origin',item['repository'])
            git('config','remote.origin.promisor','true');git('config','remote.origin.partialclonefilter','blob:none')
        present=subprocess.run(['git','cat-file','-e',item['commit']+'^{commit}'],cwd=path,capture_output=True)
        if present.returncode:git('fetch','--depth','1','--filter=blob:none','origin',item['commit'])
        git('sparse-checkout','init','--cone');git('sparse-checkout','set','cmake','commonTools')
        git('checkout','--detach',item['commit'])
        directories=read('ls-tree','-r','-d','--name-only',item['commit'],'packages').splitlines()
        metadata=[d for d in directories if d.endswith('/cmake')]
        def add(paths,label):
            for offset in range(0,len(paths),12):
                batch=paths[offset:offset+12]
                print(label,offset,len(paths),flush=True)
                git('sparse-checkout','add',*batch)
        add(metadata,'metadata')
        add(item.get('source_extra_directories',['commonTools','packages/belos/doc']),'support')
        roots=read('ls-tree','-d','--name-only',item['commit']+':packages').splitlines()
        key=lambda value:value.casefold().replace('-','').replace('_','')
        requested=item.get('source_packages',PACKAGES)
        for package in requested:
            matches=[r for r in roots if key(r)==key(package)]
            if len(matches)!=1:raise ValueError('cannot resolve Trilinos package '+package)
            prefix='packages/'+matches[0]
            allowed=sorted(d for d in directories if(d==prefix or d.startswith(prefix+'/'))and
                not any(part.casefold()in DENY for part in d.split('/')[2:]))
            parents={str(Path(d).parent)for d in allowed}
            leaves=[d for d in allowed if d not in parents]
            add(leaves,package)
        git('diff','--exit-code',item['commit'],'--')
    (BASE/'trilinos_source_selection.json').write_text(json.dumps(dict(commit=item['commit'],
        source_packages=list(requested),excluded_directory_names=sorted(DENY),git_batches=12),indent=2)+'\n')


if __name__=='__main__':
    lock=json.loads((ROOT/'physical/xyce_tools.lock.json').read_text())
    populate(BASE/'git/trilinos',lock['trilinos'])
