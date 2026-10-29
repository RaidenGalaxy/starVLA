#!/usr/bin/env python3
"""Compare actual result/exception/count traces across pipeline versions."""
import argparse,hashlib,json
from pathlib import Path
from runtime import validated_build


def check(builds,out):
    guards=[];reports=[]
    for build in builds:
        guard,caps=validated_build(build);guards.append(guard)
        path=build/'verification_events.json';report=json.loads(path.read_text())
        if not report.get('passed')or not report.get('all_formats')or report.get('test_suite_version')!=2:
            raise ValueError('all-format event verification suite 2 required')
        for name in('rtl_sha256','simulator_sha256'):
            if report[name]!=caps[name]:raise ValueError('event report refers to another build')
        for filename,key in [('event_vectors.txt','trace_sha256'),('event_simulation.log','simulation_log_sha256')]:
            if hashlib.sha256((build/filename).read_bytes()).hexdigest()!=report[key]:raise ValueError('event trace or log changed')
        reports.append(dict(build=str(build),verification=str(path),verification_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**report))
    baseline=reports[0]
    for report in reports[1:]:
        for key in('rows','cols','trace_sha256','reference_sha256','semantic_output_sha256','reads','counter_snapshots'):
            if report[key]!=baseline[key]:raise AssertionError('pipeline event trace mismatch: '+key)
    if len({r['stages']for r in reports})<2:raise ValueError('at least two pipeline versions required')
    result=dict(passed=True,scope='array_pipeline_numerical_equivalence',all_formats=True,
        includes_sticky_flags_and_event_counters=True,observation_times_compared=False,
        timing_certification=False,builds=reports)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,reads_per_build=baseline['reads'],counter_snapshots_per_build=baseline['counter_snapshots'],
        stages=[r['stages']for r in reports],semantic_output_sha256=baseline['semantic_output_sha256'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--builds',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if len(args.builds)<2:parser.error('at least two builds required')
    check([p.resolve()for p in args.builds],args.out.resolve())
