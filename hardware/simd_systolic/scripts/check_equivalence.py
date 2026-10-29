#!/usr/bin/env python3
"""Compare F2 results against the independent rational arithmetic oracle."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from runtime import validated_build
from check_alu import make_vectors

ROOT=Path(__file__).resolve().parents[1]


def check(stages,build_dirs=None,out_dir=None):
    if stages != [2]:raise ValueError('only the F2/reference comparison is supported')
    if build_dirs is not None and len(build_dirs)!=len(stages):raise ValueError('one build directory per pipeline depth is required')
    out=Path(out_dir).resolve()if out_dir else ROOT/'.build/equivalence';out.mkdir(parents=True,exist_ok=True)
    vectors,expected=make_vectors(100,True)
    path=out/'vectors.txt';path.write_text(''.join(vectors));baseline=None;builds=[];guards=[]
    for index,depth in enumerate(stages):
        dest=Path(build_dirs[index]).resolve()if build_dirs else ROOT/f'.build/alu_{depth}_1x1';guard,caps=validated_build(dest);guards.append(guard)
        if caps['target']!='alu'or caps['stages']!=depth:raise ValueError('equivalence build scope differs from requested pipeline')
        result=subprocess.run([str(dest/'sim'),str(path),'101'],capture_output=True,text=True,check=True,timeout=600)
        (out/f'profile_{depth}.log').write_text(result.stdout+result.stderr)
        actual={}
        for line in result.stdout.splitlines():
            if not line or not line[0].isdigit():continue
            tag,data,flags,overflow,illegal,*_=line.split();tag=int(tag)
            if tag in actual:raise AssertionError('duplicate response tag')
            value=(int(data,16),int(flags,16),int(overflow,16))
            if value!=expected[tag]or int(illegal):raise AssertionError(f'profile {depth}, tag {tag}: numerical mismatch')
            actual[tag]=value
        if len(actual)!=len(vectors):raise AssertionError('missing tagged responses')
        if baseline is None:baseline=actual
        elif baseline!=actual:raise AssertionError('pipeline versions disagree after tag alignment')
        builds.append(dict(stages=depth,rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256']))
    report=dict(passed=True,vectors=len(vectors),all_formats=True,random_backpressure=True,
        exhaustive_formats=['I4','F4','F8'],comparison='F2 response data/flags/overflow versus independent rational reference',
        oracle_checked=True,builds=builds,trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stages',type=int,nargs='+',default=[2])
    parser.add_argument('--build-dirs',type=Path,nargs='+');parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    check(args.stages,args.build_dirs,args.out)
