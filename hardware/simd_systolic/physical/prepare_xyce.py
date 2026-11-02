#!/usr/bin/env python3
"""Build a pinned native BSIM-CMG107 simulator in the local dependency tree.

No system installation or proprietary models are used. The source commits,
commands, compiler version, BLAS/LAPACK and final executable are recorded.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'.deps/xyce-native'
LOCK=ROOT/'physical/xyce_tools.lock.json'
DRIVER_SOURCE=Path(__file__).read_bytes()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def verify_source(name,path,commit):
    changed=subprocess.check_output(['git','diff','--name-only',commit,'--'],cwd=path,text=True).splitlines()
    generated='SuiteSparse_config/SuiteSparse_config.h'
    if name=='suitesparse'and changed==[generated]:
        original=subprocess.check_output(['git','show',commit+':'+generated],cwd=path)
        old=b'\n    #define SUITESPARSE_CONFIG_TIMER omp_get_wtime\n'
        if original.count(old)!=1:raise ValueError('unexpected upstream SuiteSparse timer definition')
        expected=original.replace(old,b'\n    #define SUITESPARSE_CONFIG_TIMER clock_gettime\n')
        if(path/generated).read_bytes()!=expected:raise ValueError('unrecognized SuiteSparse source change')
        (BASE/'suitesparse_generated_header.json').write_text(json.dumps(dict(
            source_commit=commit,path=generated,upstream_sha256=hashlib.sha256(original).hexdigest(),
            generated_sha256=hashlib.sha256(expected).hexdigest(),
            cause='Upstream CMake selects clock_gettime when OpenMP is disabled'),indent=2)+'\n')
    elif changed:raise ValueError('dependency source changed: '+name+': '+','.join(changed))


def source(name,item):
    path=BASE/'git'/name;path.mkdir(parents=True,exist_ok=True)
    if item.get('checkout_strategy')=='package_batches':
        if name!='trilinos':raise ValueError('unknown package checkout strategy')
        marker=BASE/'trilinos_source_selection.json'
        ready=json.loads(marker.read_text())if marker.exists()else{}
        if ready.get('commit')!=item['commit']or ready.get('source_packages')!=item['source_packages']or any(
                not(path/p).is_dir()for p in item.get('source_extra_directories',[])):
            from fetch_trilinos import populate
            populate(path,item)
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()!=item['commit']:
            raise ValueError('Trilinos source commit differs')
        verify_source(name,path,item['commit'])
        return path
    env={**os.environ,'GIT_TERMINAL_PROMPT':'0'}
    with(BASE/(name+'_source.log')).open('w')as log:
        def git(*args):
            return subprocess.run(['git','-c','http.version=HTTP/1.1','-c','http.lowSpeedLimit=1024',
                '-c','http.lowSpeedTime=120',*args],cwd=path,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not(path/'.git').is_dir():
            git('init');git('remote','add','origin',item['repository'])
            git('config','remote.origin.promisor','true');git('config','remote.origin.partialclonefilter','blob:none')
            git('config','core.sparseCheckout','true')
            (path/'.git/info/sparse-checkout').write_text('\n'.join(item['sparse_patterns'])+'\n')
        current=subprocess.run(['git','rev-parse','HEAD'],cwd=path,capture_output=True,text=True)
        if current.returncode or current.stdout.strip()!=item['commit']:
            git('fetch','--depth','1','--filter=blob:none','origin',item['commit']);git('checkout','--detach',item['commit'])
        if subprocess.check_output(['git','remote','get-url','origin'],cwd=path,text=True).strip()!=item['repository']:
            raise ValueError('dependency remote differs: '+name)
        verify_source(name,path,item['commit'])
    return path


def run(args):
    BASE.mkdir(parents=True,exist_ok=True);lock=json.loads(LOCK.read_text())
    driver_digest=hashlib.sha256(DRIVER_SOURCE).hexdigest()
    (BASE/('prepare_driver_'+driver_digest[:12]+'.py')).write_bytes(DRIVER_SOURCE)
    if args.phase in('sources','all'):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:
            list(pool.map(lambda entry:source(*entry),lock.items()))
        if args.phase=='sources':return
    sources={name:BASE/'git'/name for name in lock}
    needed={'suitesparse'}if args.phase=='suitesparse'else set(sources)
    for name in needed:
        path=sources[name]
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
        if commit!=lock[name]['commit']:raise ValueError('dependency commit differs: '+name)
        verify_source(name,path,commit)
    install=BASE/'install';suite=install/'suitesparse';tri=install/'trilinos';xyce=install/'xyce'
    env={**os.environ,'LD_LIBRARY_PATH':str(suite/'lib')+':'+os.environ.get('LD_LIBRARY_PATH','')}
    blas=Path('/lib/x86_64-linux-gnu/libblas.so.3');lapack=Path('/lib/x86_64-linux-gnu/liblapack.so.3')
    if not blas.is_file()or not lapack.is_file():raise ValueError('BLAS/LAPACK runtime libraries missing')
    record=BASE/'build_commands.jsonl'
    def call(name,command):
        print(name,flush=True)
        with record.open('a')as stream:stream.write(json.dumps(dict(name=name,command=list(map(str,command)),driver_sha256=driver_digest))+'\n')
        log_path=BASE/(name+'.log')
        if log_path.exists():shutil.copy2(log_path,BASE/(name+'_'+sha(log_path)[:12]+'.log'))
        with log_path.open('w')as log:
            subprocess.run(list(map(str,command)),env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if args.phase in('suitesparse','all'):
        directory=BASE/'build/suitesparse'
        call('configure_suitesparse',['cmake','-S',sources['suitesparse'],'-B',directory,
            '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_INSTALL_PREFIX='+str(suite),
            '-DSUITESPARSE_ENABLE_PROJECTS=suitesparse_config;amd','-DSUITESPARSE_USE_OPENMP=OFF',
            '-DBLAS_LIBRARIES='+str(blas),'-DLAPACK_LIBRARIES='+str(lapack)])
        call('build_suitesparse',['cmake','--build',directory,'-j',args.jobs,'--target','install'])
    if args.phase in('trilinos','all'):
        directory=BASE/'build/trilinos'
        call('configure_trilinos',['cmake','-S',sources['trilinos'],'-B',directory,
            '-C',sources['xyce']/'cmake/trilinos/trilinos-base.cmake',
            '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_STANDARD=17','-DCMAKE_INSTALL_PREFIX='+str(tri),
            '-DTrilinos_ENABLE_Fortran=OFF','-DTPL_ENABLE_MPI=OFF','-DTrilinos_ENABLE_TESTS=OFF',
            '-DTrilinos_ENABLE_EXAMPLES=OFF','-DTPL_AMD_INCLUDE_DIRS='+str(suite/'include/suitesparse'),
            '-DTPL_AMD_LIBRARIES='+str(suite/'lib/libamd.so')+';'+str(suite/'lib/libsuitesparseconfig.so'),
            '-DTPL_BLAS_LIBRARIES='+str(blas),'-DTPL_LAPACK_LIBRARIES='+str(lapack)])
        call('build_trilinos',['cmake','--build',directory,'-j',args.jobs,'--target','install'])
    if args.phase in('xyce','all'):
        if not(tri/'lib/cmake/Trilinos/TrilinosConfig.cmake').is_file():
            raise ValueError('complete --phase trilinos successfully before building Xyce')
        directory=BASE/'build/xyce'
        call('configure_xyce',['cmake','-S',sources['xyce'],'-B',directory,
            '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_STANDARD=17','-DCMAKE_INSTALL_PREFIX='+str(xyce),
            '-DTrilinos_ROOT='+str(tri),'-DXyce_USE_CURL=OFF','-DXyce_PLUGIN_SUPPORT=OFF',
            '-DXyce_NONFREE_MODELS=OFF','-DXyce_RAD_MODELS=OFF','-DXyce_NEURON_MODELS=OFF',
            '-DXyce_ADMS_MODELS=ON','-DXyce_ADMS_SENSITIVITIES=OFF','-DXyce_USE_FFT=OFF',
            '-DXyce_TEST_BINARIES=OFF'])
        call('build_xyce',['cmake','--build',directory,'-j',args.jobs,'--target','install'])
        executable=xyce/'bin/Xyce'
        capabilities=subprocess.run([str(executable),'-capabilities'],env=env,capture_output=True,text=True,check=True)
        (BASE/'capabilities.txt').write_text(capabilities.stdout+capabilities.stderr)
        result=dict(executable=str(executable),executable_sha256=sha(executable),
            sources=lock,lock_sha256=sha(LOCK),driver_sha256=driver_digest,
            blas_sha256=sha(blas),lapack_sha256=sha(lapack),
            compiler=subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
            generated_suitesparse_header_sha256=sha(BASE/'suitesparse_generated_header.json')if(BASE/'suitesparse_generated_header.json').exists()else None,
            native_model='BSIM-CMG107 level107; no OSDI compiler port',signoff_toolchain=False)
        (BASE/'capabilities.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['sources','suitesparse','trilinos','xyce','all'],default='all')
    p.add_argument('--jobs',type=int,default=8);args=p.parse_args()
    if not 1<=args.jobs<=16:p.error('jobs must be within1..16')
    run(args)
