#!/usr/bin/env python3
"""Remove unobservable mapped I64 cells, certify structure, then check numerics."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from run import ROOT,sha
from observable_cone import certify
from format_mapped_verilog import wrap_assignments


def run(args):
    source=args.source.resolve();out=args.out.resolve()
    identity=json.loads((source/'build_identity.json').read_text())
    parent=source/'check_mapped/verification.json';numerical=json.loads(parent.read_text())
    if (identity['scope'],identity['stages'],identity['analysis_period_ns'])!=('int64',2,1):
        raise ValueError('requires original mapped F2 I64 at1ns')
    if not numerical['passed']or numerical['netlist_sha256']!=sha(source/'synth.v'):
        raise ValueError('parent has no matching mapped numerical proof')
    if sha(identity['liberty_file'])!=identity['liberty_sha256']:raise ValueError('process library changed')
    if out.exists():raise ValueError('preserve previous cleanup experiments')
    out.mkdir(parents=True);shutil.copytree(source/'inputs',out/'inputs')
    if sha(source/'mapping.lib')!=identity['abc_library_sha256']:raise ValueError('parent ABC library changed')
    shutil.copy2(source/'mapping.lib',out/'mapping.lib')
    shutil.copy2(__file__,out/'cleanup_driver.py')
    shutil.copy2(ROOT/'physical/observable_cone.py',out/'observable_cone_driver.py')
    script=out/'cleanup.ys';script.write_text(f'''read_liberty -lib {identity['liberty_file']}
read_verilog {source/'synth.v'}
hierarchy -check -top SimdInt64
write_json {out/'before.json'}
opt_clean -purge
check -assert
tee -o {out/'cell_statistics.json'} stat -json -liberty {identity['liberty_file']}
write_verilog -noattr -noexpr {out/'synth.v'}
write_json {out/'after.json'}
''')
    with(out/'cleanup.log').open('w')as stream:
        subprocess.run(['yosys','-Q','-T','-s',str(script)],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=300)
    shutil.copy2(out/'cleanup.log',out/'synth.log')
    result=certify(json.loads((out/'before.json').read_text()),json.loads((out/'after.json').read_text()),'SimdInt64')
    result.update(source=str(source),source_identity_sha256=sha(source/'build_identity.json'),
        source_mapped_sha256=sha(source/'synth.v'),mapped_sha256=sha(out/'synth.v'),
        source_numerical_verification=str(parent),source_numerical_sha256=sha(parent),
        pipeline_stages=2,core_period_ns=1,physical_frequency_certified=False,
        evidence={str(p):sha(p)for p in [parent,source/'synth.v',out/'before.json',out/'after.json',
            out/'synth.v',out/'cleanup.log',script,out/'cleanup_driver.py',out/'observable_cone_driver.py']})
    proof=out/'observable_cleanup.json';proof.write_text(json.dumps(result,indent=2)+'\n')
    for key in list(identity):
        if key.startswith('tool_input_')or key in('clock_options','resumed_stage','resumed_stage_manifest_sha256'):
            identity.pop(key)
    identity.update(mapped_sha256=sha(out/'synth.v'),observable_cleanup_manifest=str(proof),
        observable_cleanup_manifest_sha256=sha(proof))
    shutil.copy2(out/'synth.v',out/'tool_input.v');wrapped=wrap_assignments(out/'tool_input.v',all_statements=True)
    shutil.copy2(ROOT/'physical/format_mapped_verilog.py',out/'netlist_formatter.py')
    identity.update(tool_input_netlist_sha256=sha(out/'tool_input.v'),
        tool_input_formatter_sha256=sha(out/'netlist_formatter.py'),tool_input_wrapped_lines=wrapped,
        tool_input_transform='whitespace only; identical non-whitespace token sequence')
    (out/'build_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'physical/check_int64.py'),str(out),'--mapped','--count','1000'],check=True,cwd=ROOT,timeout=1800)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);run(parser.parse_args())
