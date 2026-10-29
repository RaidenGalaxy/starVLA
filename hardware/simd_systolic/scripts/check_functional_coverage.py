#!/usr/bin/env python3
"""Observed component numerical coverage; protocol/system acceptance is separate.

Freeze the entire bin catalogue BEFORE generating any input or running the
simulator. Missing bins remain visible and force failure. No predeclared
stimulus is counted without matching an actual DUT result.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from runtime import validated_build
from simd_eval.reference import Format,Op,WIDTHS,ACC_WIDTHS,ENCODINGS,operate,pack,to_bits
from simd_eval.functional_coverage import FunctionalCoverage,numerical_bins,mask_bins


def retain_report(out,report,caps):
    (out/'functional_coverage.json').write_text(json.dumps(report,indent=2)+'\n')
    with(out/'functional_coverage.csv').open('w',newline='')as stream:
        writer=csv.writer(stream);writer.writerow(['bin','hits','first_witness'])
        model=json.loads((out/'functional_model.json').read_text())
        for item in model['bins']:
            name=item['id'];writer.writerow([name,report['hits'].get(name,0),json.dumps(report['first_witnesses'].get(name))])
    code=out/'coverage.dat'
    if caps.get('code_coverage_instrumented'):
        if not code.is_file():raise AssertionError('instrumented simulator did not emit code coverage')
        (out/'coverage.dat.identity.json').write_text(json.dumps(dict(
            rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'],
            coverage_sha256=hashlib.sha256(code.read_bytes()).hexdigest(),
            functional_run_passed=report['passed']),indent=2)+'\n')


def vectors():
    requests=[];state=[0]*16
    def emit(fmt,op,a,b,mask=65535,ctx=0,initial=0,clear=False,labels=(),preload=True):
        acc=initial if preload else 0 if clear else state[ctx]
        result=operate(a,b,acc,fmt,op,mask);state[ctx]=result[0]
        requests.append(dict(fmt=int(fmt),op=int(op),a=a,b=b,mask=mask,ctx=ctx,
                             acc=acc,clear=clear,preload=preload,expected=result,labels=list(labels)))
    for fmt in Format:
        w,aw=WIDTHS[fmt],ACC_WIDTHS[fmt];lanes=64//w
        a=pack([to_bits(1,fmt)if fmt>=Format.F4 else 3]*lanes,w)
        b=pack([to_bits(-.5,fmt)if fmt>=Format.F4 else 2]*lanes,w)
        dst=Format.F64 if fmt==Format.F64 else Format.F32
        acc=pack([to_bits(.25,dst)if fmt>=Format.F4 else 7]*lanes,aw)
        for op in range(8 if fmt<Format.F4 else 4):
            for ctx in range(16):emit(fmt,op,a,b,ctx=ctx,initial=acc)
        for name,mask in mask_bins(lanes).items():emit(fmt,Op.MAC,a,b,mask=mask,initial=acc)
    # Each signed nibble pair visits each independent lane with other nibbles
    # deliberately nonzero, exposing accidental packed-byte cross products.
    for lane in range(16):
        for a in range(16):
            for b in range(16):
                aa=pack([((a+3*i)&15)for i in range(16)],4)
                bb=pack([((b+5*i)&15)for i in range(16)],4)
                aa=(aa&~(15<<(4*lane)))|(a<<(4*lane))
                bb=(bb&~(15<<(4*lane)))|(b<<(4*lane))
                emit(Format.I4,Op.MAC,aa,bb,mask=1<<lane,ctx=(a+b)%16,clear=True,preload=False)
                requests[-1]['pair_sweep']=True
    for a in range(16):
        for b in range(16):
            emit(Format.I4,Op.MAC,pack([(a+3*i)&15 for i in range(16)],4),
                 pack([(b+5*i)&15 for i in range(16)],4),clear=True,preload=False)
            requests[-1]['pair_sweep']=True
    for fmt in list(Format)[:5]:
        w,aw=WIDTHS[fmt],ACC_WIDTHS[fmt];lanes=64//w
        for negative in (False,True):
            initial=1<<(aw-1)if negative else(1<<(aw-1))-1
            a=(1<<w)-1 if negative else 1
            emit(fmt,Op.MAC,pack([a]*lanes,w),pack([1]*lanes,w),
                 initial=pack([initial]*lanes,aw),labels=[f'overflow/{fmt.name}/{"negative"if negative else"positive"}'])
    for fmt in list(Format)[5:]:
        w,aw=WIDTHS[fmt],ACC_WIDTHS[fmt];lanes=64//w
        dst=Format.F64 if fmt==Format.F64 else Format.F32
        one=to_bits(1,fmt);zero=0;minus=to_bits(-1,fmt)
        e,m=ENCODINGS[dst];inf=((1<<e)-1)<<m
        def floating(a,b,acc=0,category=None):
            emit(fmt,Op.MAC,pack([a]*lanes,w),pack([b]*lanes,w),initial=pack([acc]*lanes,aw),
                 labels=[f'float/{fmt.name}/{category}']if category else[])
        floating(zero,one,category='zero')
        floating(1<<(w-1),one,1<<(aw-1),category='negative_zero')
        floating(1,one,category='subnormal')
        floating(one,one,to_bits(-1,dst),category='cancellation')
        exponent=53 if fmt==Format.F64 else 24
        floating(one,one,to_bits(1<<exponent,dst),category='tie_even')
        floating(one,one,to_bits((1<<exponent)+2,dst),category='tie_odd')
        floating(one,one,inf|1,category='acc_nan')
        if fmt!=Format.F4:
            se,sm=ENCODINGS[fmt];source_inf=((1<<se)-1)<<sm
            floating(source_inf|1,one,category='input_nan')
            floating(source_inf,one,category='input_inf')
        if fmt in (Format.BF16,Format.F32,Format.F64):
            floating(source_inf-1,to_bits(2,fmt))
            floating(1,1 if fmt==Format.BF16 else to_bits(.5,fmt))
    for fmt in (Format.I32,Format.I64):
        w=WIDTHS[fmt];lanes=64//w;a=pack([1<<(w-1)]*lanes,w)
        for k in range(17):
            emit(fmt,Op.MAC,a,a,ctx=0,clear=k==0,preload=False)
            requests[-1]['wide_chain']=True
    return requests


def observed_bins(q):
    fmt=Format(q['fmt']);lanes=64//WIDTHS[fmt]
    ids=[f'op/{fmt.name}/{q["op"]}/{q["ctx"]}/0/0/{lane}'for lane in range(lanes)if q['mask']>>lane&1]
    ids.extend(f'mask/{fmt.name}/{name}'for name,value in mask_bins(lanes).items()if value==(q['mask']&((1<<lanes)-1)))
    if fmt==Format.I4 and q['op']==int(Op.MAC):
        ids.extend(f'int4/{lane}/{q["a"]>>(4*lane)&15}/{q["b"]>>(4*lane)&15}'for lane in range(16)if q['mask']>>lane&1)
    if fmt in(Format.I32,Format.I64)and q['op']==int(Op.MAC):
        aw=ACC_WIDTHS[fmt]
        if (q['expected'][0]&((1<<aw)-1)).bit_length()>2*WIDTHS[fmt]and not(q['expected'][0]>>(aw-1)&1):ids.append(f'wide/{fmt.name}')
    if fmt>=Format.F4:
        for flag,bit in [('NV',16),('NX',1)]+([('OF',4),('UF',2)]if fmt in(Format.BF16,Format.F32,Format.F64)else[]):
            if q['expected'][1]&bit:ids.append(f'flag/{fmt.name}/{flag}')
    return ids+q['labels']


def check_array(build_dir,out,caps,timeout):
    from simd_eval.cycle_model import mesh_trace_cycles,verify_mesh_cycles
    identity={k:caps[k]for k in('target','stages','rows','cols','rtl_sha256','simulator_sha256')}
    identity['checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    coverage=FunctionalCoverage(identity,numerical_bins('array',caps['rows'],caps['cols']),
        exclusions=['Array exposes MAC only; independent PE covers other operations',
                    'No division; finite FP4 source has no NaN/Inf; narrow FP32 accumulation OF/UF applicability follows destination range'])
    model=out/'functional_model.json'
    if model.exists()and json.loads(model.read_text())!=coverage.model:raise ValueError('refuse to replace frozen coverage model')
    model.write_text(json.dumps(coverage.model,indent=2)+'\n')
    cases=[q for q in vectors()if q['op']==int(Op.MAC)and not q.get('pair_sweep')]
    for a in range(16):
        for b in range(16):
            cases.append(dict(fmt=0,op=1,a=a,b=pack([(b+5*i)&15 for i in range(16)],4),
                              mask=65535,ctx=(a+b)%16,acc=0,clear=True,preload=True,labels=[]))
    commands=[];expected={};observations={};events={};tag=0;state={};expected_macs=0
    totals=[0,0,0,0]
    rows,cols=caps['rows'],caps['cols']
    for q in cases:
        fmt=Format(q['fmt']);w=WIDTHS[fmt];lanes=64//w;ctx=q['ctx']
        commands.append('D\n')
        if q['preload']:
            for r in range(rows):
                for c in range(cols):
                    commands.append(f'S {ctx} {r} {c} 0 {q["acc"]:0128x}\n');state[ctx,r,c]=(q['acc'],0)
        aa=[q['a']&((1<<w)-1)for _ in range(rows)];bb=[q['b']for _ in range(cols)]
        commands.append(f'I {int(fmt)} {ctx} {int(q["clear"])} {tag} {(1<<rows)-1:x} '+
                        ' '.join(f'{x:x}'for x in [q['mask']]*cols+aa+bb)+'\n');tag+=1
        expected_macs+=rows*cols*(q['mask']&((1<<lanes)-1)).bit_count()
        commands.append('D\n')
        for r in range(rows):
            for c in range(cols):
                old,sticky=(0,0)if q['clear']else state.get((ctx,r,c),(0,0))
                qa=dict(q,a=pack([aa[r]]*lanes,w),b=bb[c],acc=old)
                qa['expected']=operate(qa['a'],qa['b'],old,fmt,Op.MAC,q['mask'])
                value,flags,overflow=qa['expected'];state[ctx,r,c]=(value,sticky|flags)
                totals[0]+=(q['mask']&((1<<lanes)-1)).bit_count();totals[1]+=1
                totals[2]+=overflow.bit_count();totals[3]+=int(bool(flags))
                expected[tag]=(value,sticky|flags);observations[tag]=(qa,r,c)
                commands.append(f'P {ctx} {r} {c} {tag}\n');tag+=1
        commands.append(f'E {tag}\n');events[tag]=tuple(totals);tag+=1
    trace=out/'commands.txt';trace.write_text(''.join(commands));log=out/'simulation.log'
    env=os.environ.copy()
    if caps.get('code_coverage_instrumented'):env['SIMD_CODE_COVERAGE']=str(out/'coverage.dat')
    with log.open('w')as stream:run=subprocess.run([str(build_dir/'sim'),str(trace)],stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=timeout)
    if run.returncode:raise RuntimeError(f'array simulator failed: {log}')
    stdout=log.read_text();cycles=verify_mesh_cycles(stdout,mesh_trace_cycles(caps,commands));seen=set();event_seen=set();macs=0
    for line in stdout.splitlines():
        if line.startswith('E '):
            fields=line.split();event_tag=int(fields[1])
            if event_tag in event_seen or tuple(map(int,fields[3:]))!=events[event_tag]:raise AssertionError('array exception/overflow event mismatch')
            event_seen.add(event_tag);continue
        if not line.startswith('R '):continue
        _,tag,data,flags,cycle,mac,commits=line.split();tag=int(tag)
        if tag in seen or tag not in expected:raise AssertionError('duplicate/unexpected array response')
        q,r,c=observations[tag]
        ids=[]
        for name in observed_bins(q):
            if name.startswith('op/'):
                parts=name.split('/');parts[4:6]=[str(r),str(c)];name='/'.join(parts)
            ids.append(name)
        # Readback exposes sticky flags, while exception/overflow events have
        # an independently checked counter suite; labels assert arithmetic.
        coverage.checked(ids,witness=dict(read_tag=tag,row=r,col=c,observed_cycle=int(cycle)),
                         actual=(int(data,16),int(flags)),expected=expected[tag])
        seen.add(tag);macs=int(mac)
    if len(seen)!=len(expected)or macs!=expected_macs or event_seen!=events.keys():raise AssertionError('array completion/MAC accounting mismatch')
    report=coverage.report();report.update(reads=len(seen),macs=macs,independent_cycle_checks=cycles,
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        component_numerical_subset=True,complete_configuration_accepted=False,
        limitation='Event flags/overflow counters, long-K, random protocols and system checks remain separate required acceptance groups.')
    retain_report(out,report,caps)
    print(json.dumps({k:v for k,v in report.items()if k not in('hits','first_witnesses')},indent=2))
    if not report['passed']:raise AssertionError('array coverage has missing critical bins')
    return report


def check(build_dir,out,timeout=1800):
    build_dir=Path(build_dir).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    guard,caps=validated_build(build_dir)
    (out/'checker_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out/'reference_snapshot.py').write_bytes((ROOT/'python/simd_eval/reference.py').read_bytes())
    if caps['target']=='array':return check_array(build_dir,out,caps,timeout)
    if caps['target']not in('pe','alu'):raise ValueError('numerical component checker requires PE/ALU/array')
    identity={k:caps[k]for k in('target','stages','rows','cols','rtl_sha256','simulator_sha256')}
    identity['checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    coverage=FunctionalCoverage(identity,numerical_bins(caps['target']),exclusions=[
        'No division operation: DZ is not an applicable event',
        'FP4 source encoding is finite; FP4 accumulator still supports IEEE FP32 NaNs',
        'FP4/FP8/FP16 finite products with same-order FP32 accumulation cannot generate OF/UF; NX/NV remain covered'])
    model=out/'functional_model.json'
    if model.exists()and json.loads(model.read_text())!=coverage.model:raise ValueError('refuse to replace frozen coverage model')
    model.write_text(json.dumps(coverage.model,indent=2)+'\n')
    cases=vectors();lines=[]
    for tag,q in enumerate(cases):
        if caps['target']=='pe':
            line=f'{q["fmt"]} {q["op"]} {q["a"]:x} {q["b"]:x} {q["mask"]:x} {q["ctx"]} {int(q["clear"])} {tag} {q["expected"][0]:x} {q["expected"][1]}'
            if q['preload']:line+=f' {q["acc"]:x} 0'
        else:line=f'{q["fmt"]} {q["op"]} {q["a"]:x} {q["b"]:x} {q["acc"]:x} {q["mask"]:x} {q["ctx"]} {tag}'
        lines.append(line+'\n')
    trace=out/'vectors.txt';trace.write_text(''.join(lines))
    env=os.environ.copy()
    if caps.get('code_coverage_instrumented'):env['SIMD_CODE_COVERAGE']=str(out/'coverage.dat')
    command=[str(build_dir/'sim'),str(trace)]+(['101']if caps['target']=='alu'else[])
    log=out/'simulation.log'
    with log.open('w')as stream:run=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=timeout)
    if run.returncode:
        (out/'failure.json').write_text(json.dumps(dict(status='simulator_failed',returncode=run.returncode,identity=identity,log=str(log)),indent=2)+'\n')
        raise RuntimeError(f'simulator failed: {log}')
    seen=set();contexts={};admissions={}
    for line in log.read_text().splitlines():
        fields=line.split()
        if fields[:1]==['U']:contexts[int(fields[1])]=int(fields[2]);continue
        if fields[:1]==['J']:admissions[int(fields[1])]=int(fields[2]);continue
        if caps['target']=='pe':
            if fields[:1]!=['O']:continue
            _,tag,data,flags,overflow,begin,end=fields
            tag=int(tag);actual=(int(data,16),int(flags),int(overflow))
        else:
            if not fields or not fields[0].isdigit():continue
            tag,data,flags,overflow,illegal,begin,end=fields;tag=int(tag)
            if int(illegal):raise AssertionError('legal coverage operation marked illegal')
            actual=(int(data,16),int(flags,16),int(overflow,16))
        if tag in seen or not 0<=tag<len(cases):raise AssertionError('duplicate/unexpected response')
        q=cases[tag]
        coverage.checked(observed_bins(q),witness=dict(tag=tag,accepted_cycle=int(begin),response_cycle=int(end)),actual=actual,expected=q['expected'])
        seen.add(tag)
    if len(seen)!=len(cases):raise AssertionError('missing observed responses')
    if caps['target']=='pe':
        if set(contexts)!=seen or set(admissions)!=seen:raise AssertionError('missing actual admission/context events; rebuild with the new harness')
        if any(contexts[tag]!=cases[tag]['ctx']for tag in seen):raise AssertionError('response context mismatch')
    report=coverage.report();report.update(vectors=len(cases),trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),component_numerical_subset=True,
        complete_configuration_accepted=False,limitation='Protocol, long-K, topology and system coverage are separate required acceptance groups.')
    retain_report(out,report,caps)
    print(json.dumps({k:v for k,v in report.items()if k not in('hits','first_witnesses')},indent=2))
    if not report['passed']:raise AssertionError('coverage has missing critical bins; see functional_coverage.json')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build-dir',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path);parser.add_argument('--timeout',type=int,default=1800)
    args=parser.parse_args();check(args.build_dir,args.out,args.timeout)
