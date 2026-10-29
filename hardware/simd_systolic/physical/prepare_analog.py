#!/usr/bin/env python3
"""Prepare a local, pinned ngspice/OpenVAF BSIM-CMG107 diagnostic toolchain.

The upstream Berkeley equations and ASAP7 parameter values stay unchanged.
The compiler port only moves/removes Xyce annotations and spells a lower bound
as 1e-10 instead of 0.1n, avoiding a one-ULP floating-point boundary mismatch.
"""
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'.deps/analog'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def port_annotations(text):
    lines=[]
    for line in text.splitlines(keepends=True):
        if line.startswith('`define attr('):continue
        instance=False
        while '`attr('in line:
            start=line.index('`attr(');i=start+6;depth=1;quoted=False
            while depth:
                char=line[i]
                if char=='"'and(i==0 or line[i-1]!='\\'):quoted=not quoted
                if not quoted:
                    if char=='(':depth+=1
                    if char==')':depth-=1
                i+=1
            instance|='type="instance"'in line[start+6:i-1]
            line=line[:start]+line[i:]
        if instance:
            if 'parameter 'not in line:raise ValueError('unexpected instance annotation')
            line=re.sub(r'^(\s*)parameter ',r'\1(* type="instance" *) parameter ',line)
        lines.append(line)
    return ''.join(lines)


def download(url,path,digest):
    if path.exists():
        if sha(path)!=digest:raise ValueError('cached dependency changed: '+str(path))
        return
    temporary=path.with_suffix(path.suffix+'.part')
    try:
        with urllib.request.urlopen(url,timeout=60)as response,temporary.open('wb')as stream:
            shutil.copyfileobj(response,stream)
        if sha(temporary)!=digest:raise ValueError('download identity differs: '+url)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare():
    lock_path=ROOT/'physical/analog_tools.lock.json';lock=json.loads(lock_path.read_text())
    downloads=BASE/'downloads';downloads.mkdir(parents=True,exist_ok=True)
    for name in('ngspice','openvaf'):
        item=lock[name];archive=downloads/item['archive'];download(item['url'],archive,item['sha256'])
        executable=BASE/item['executable']
        if not executable.exists():
            if name=='ngspice':subprocess.run(['dpkg-deb','-x',str(archive),str(BASE/'ngspice')],check=True)
            else:
                with tarfile.open(archive)as source:source.extractall(BASE/'openvaf',filter='data')
        if sha(executable)!=item['executable_sha256']:raise ValueError('tool executable changed: '+name)
    original=BASE/'bsimcmg107';original.mkdir(exist_ok=True)
    for entry in lock['bsim_cmg_107']['files']:
        path=original/entry['name'];download(entry['url'],path,entry['sha256'])
        content=path.read_bytes()
        if hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()!=entry['sha']:
            raise ValueError('upstream Git blob changed')
    port=BASE/'bsimcmg107_openvaf';port.mkdir(exist_ok=True);inputs={}
    for entry in lock['bsim_cmg_107']['files']:
        name=entry['name']
        if Path(name).suffix not in('.va','.include'):continue
        text=port_annotations((original/name).read_text())
        if name=='bsimcmg_body.include':
            old='parameter real EOTACC    =  EOT from [0.1n:inf)'
            if text.count(old)!=1:raise ValueError('unexpected BSIM-CMG lower-bound declaration')
            text=text.replace(old,'parameter real EOTACC    =  EOT from [1e-10:inf)')
        destination=port/name;destination.write_text(text);inputs[name]=sha(destination)
    output=port/'bsimcmg.osdi';manifest=BASE/'capabilities.json'
    prior=json.loads(manifest.read_text())if manifest.exists()else{}
    if not(output.exists()and prior.get('port_sources_sha256')==inputs and
            prior.get('compiler_sha256')==lock['openvaf']['executable_sha256']and
            prior.get('osdi_sha256')==sha(output)):
        with(port/'compile.log').open('w')as log:
            subprocess.run([str(BASE/lock['openvaf']['executable']),str(port/'bsimcmg.va'),'-o',str(output)],
                stdout=log,stderr=subprocess.STDOUT,check=True)
    result=dict(ngspice=str(BASE/lock['ngspice']['executable']),
        ngspice_sha256=lock['ngspice']['executable_sha256'],osdi=str(output),osdi_sha256=sha(output),
        compiler_sha256=lock['openvaf']['executable_sha256'],port_sources_sha256=inputs,
        lock_sha256=sha(lock_path),preparation_driver_sha256=sha(__file__),
        model_version='BSIM-CMG107.0.0',equations_modified=False,
        compiler_port='Xyce metadata adapted; equivalent decimal lower-bound spelling; original sources retained',
        signoff_toolchain=False)
    manifest.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    result=prepare();print(json.dumps({k:v for k,v in result.items()if k!='port_sources_sha256'},indent=2))
