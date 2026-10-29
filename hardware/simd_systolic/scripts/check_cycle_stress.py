#!/usr/bin/env python3
"""Randomize legal mesh timing, including empty waves and mixed observation ports."""
import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from runtime import validated_build
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from simd_eval.cycle_model import mesh_trace_cycles,verify_mesh_cycles


def check(build,out,batches=40):
    guard,caps=validated_build(build)
    if caps['target']!='array':raise ValueError('array build required')
    out.mkdir(parents=True,exist_ok=True);rng=random.Random(52719)
    rows,cols=caps['rows'],caps['cols']
    # The minimal reproducer catches missing intermediate hop valids before
    # random waits or active PE state can hide the early busy release.
    commands=['I 4 0 1 0 0 '+' '.join(['1']*(cols+rows+cols))+'\n','D\n','E 0\n']
    tag=1;marker=1;empty_waves=1
    def read():
        nonlocal tag
        commands.append(f'{rng.choice(["R","P"])} {rng.randrange(16)} {rng.randrange(rows)} {rng.randrange(cols)} {tag}\n')
        tag+=1
    for batch in range(batches):
        # No context is reused before the preceding D. Within each batch
        # independent contexts permit back-to-back waves without scoreboarding.
        for ctx in range(rng.randrange(1,17)):
            mask=0 if batch%4==0 else rng.randrange(1<<rows)
            empty_waves+=int(mask==0)
            payload=[rng.randrange(2)for _ in range(cols)]+[1]*rows+[1]*cols
            commands.append(f'I 4 {ctx} 1 {tag} {mask:x} '+ ' '.join(f'{v:x}'for v in payload)+'\n');tag+=1
            for _ in range(rng.randrange(3)):read()
            commands.append(f'W {rng.randrange(25)}\n')
            commands.append(f'E {marker}\n');marker+=1
        commands.append('D\n')
        for _ in range(rng.randrange(1,10)):read()
        commands.append(f'T {marker}\n');marker+=1
        commands.append('D\n')
        commands.append(f'S {rng.randrange(16)} {rng.randrange(rows)} {rng.randrange(cols)} 0 0\n')
        commands.append('D\n')
    prediction=mesh_trace_cycles(caps,commands)
    trace=out/'commands.txt';trace.write_text(''.join(commands))
    actual=subprocess.run([str(build/'sim'),str(trace)],text=True,capture_output=True,timeout=1800)
    log=out/'simulation.log';log.write_text(actual.stdout+actual.stderr)
    if actual.returncode:raise RuntimeError('cycle stress simulator failed; inspect '+str(log))
    try:counts=verify_mesh_cycles(actual.stdout,prediction)
    except AssertionError as error:
        failure=dict(passed=False,status='cycle_mismatch',scope='randomized_standalone_mesh_cycles',
                     error=str(error),rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'],
                     trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                     simulation_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                     model_sha256=hashlib.sha256((ROOT/'python/simd_eval/cycle_model.py').read_bytes()).hexdigest())
        (out/'verification_failed.json').write_text(json.dumps(failure,indent=2)+'\n')
        raise
    report=dict(passed=True,scope='randomized_standalone_mesh_cycles',**counts,
                batches=batches,empty_row_waves=empty_waves,mixed_blocking_and_pipelined_reads=True,
                reads_during_compute=True,seed_and_drain=True,arithmetic_validation=False,
                stages=caps['stages'],rows=rows,cols=cols,rtl_sha256=caps['rtl_sha256'],
                simulator_sha256=caps['simulator_sha256'])
    for name,path in [('trace',trace),('simulation_log',log),('model',ROOT/'python/simd_eval/cycle_model.py'),
                      ('checker',Path(__file__))]:report[name+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--batches',type=int,default=40)
    args=parser.parse_args()
    if args.batches<1:parser.error('batches must be positive')
    check(args.build_dir.resolve(),args.out.resolve(),args.batches)
