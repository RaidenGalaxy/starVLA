#!/usr/bin/env python3
"""Check every recorded array observation against an independent wave model."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from runtime import validated_build
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from simd_eval.cycle_model import mesh_trace_cycles,verify_mesh_cycles


def check(build,positive_regression=None,events=False):
    guard,caps=validated_build(build)
    if events and positive_regression:raise ValueError('choose matrix positive regression or events')
    numeric_path=positive_regression or build/('verification_events.json'if events else'verification.json')
    proof=json.loads(numeric_path.read_text())
    if positive_regression:
        if not(proof.get('passed')and proof.get('independent_arithmetic_oracle')and proof.get('all_formats')):
            raise ValueError('successful independent positive regression required')
        candidates=[entry for entry in proof['results']if all(entry.get(k)==caps[k]
                    for k in('stages','rtl_sha256','simulator_sha256'))]
        if len(candidates)!=1:raise ValueError('positive regression does not uniquely match build')
        entry=candidates[0];numeric=dict(passed=True,**entry,**entry['observations'])
        trace=numeric_path.parent/f'matrix_{caps["stages"]}.txt'
        log=numeric_path.parent/f'matrix_{caps["stages"]}.log'
        for path,digest in [(trace,entry['trace_sha256']),(log,entry['output_sha256'])]:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('positive evidence changed: '+str(path))
    else:
        trace=build/('event_vectors.txt'if events else'array_commands.txt')
        log=build/('event_simulation.log'if events else'array_simulation.log');numeric=proof
        for path,key in [(trace,'trace_sha256'),(log,'simulation_log_sha256')]:
            if key in numeric and hashlib.sha256(path.read_bytes()).hexdigest()!=numeric[key]:
                raise ValueError('numeric evidence changed: '+str(path))
    if not numeric.get('passed')or any(numeric.get(k)!=caps[k]for k in('rtl_sha256','simulator_sha256')):
        raise ValueError('matching positive matrix regression required')
    prediction=mesh_trace_cycles(caps,trace.read_text().splitlines())
    counts=verify_mesh_cycles(log.read_text(),prediction)
    if counts['reads']!=numeric['reads']or('cycles'in numeric and counts['last_observation_cycle']!=numeric['cycles']):
        raise AssertionError('cycle log differs from numeric regression')
    # Demonstrate that one-cycle errors cannot pass this comparison.
    first=next(iter(prediction['reads']))
    bad={**prediction,'reads':{**prediction['reads'],first:prediction['reads'][first]+1}}
    try:verify_mesh_cycles(log.read_text(),bad)
    except AssertionError:pass
    else:raise AssertionError('one-cycle perturbation was not detected')
    report=dict(passed=True,scope='standalone_mesh_independent_cycle_model',**counts,
                stages=caps['stages'],rows=caps['rows'],cols=caps['cols'],
                all_formats=True,latencies_from_capabilities=True,
                observed_cycles_used_to_predict=False,one_cycle_perturbation_rejected=True,
                dma_dram_schedule_model=False,rtl_sha256=caps['rtl_sha256'],
                simulator_sha256=caps['simulator_sha256'])
    for name,path in [('trace',trace),('simulation_log',log),('numeric_report',numeric_path),
                      ('model',ROOT/'python/simd_eval/cycle_model.py'),('checker',Path(__file__))]:
        report[name+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (build/('verification_event_cycles.json'if events else'verification_cycles.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--positive-regression',type=Path,help='authenticated cached host-bridge positive regression')
    parser.add_argument('--events',action='store_true',help='check seeded boundary-event reads and counter snapshots')
    args=parser.parse_args();check(args.build_dir.resolve(),args.positive_regression.resolve()if args.positive_regression else None,args.events)
