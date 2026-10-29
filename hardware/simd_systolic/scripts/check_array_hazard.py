#!/usr/bin/env python3
"""Check scheduled context reuse and reject an invalid early mesh wave."""
import argparse,hashlib,json,resource,subprocess
from pathlib import Path
from runtime import validated_build


def check(build):
    guard,caps=validated_build(build)
    if caps['target']!='array'or not caps.get('scheduled_streaming_admission')or not caps.get('inline_chisel_assertions'):
        raise ValueError('scheduled streaming array with embedded Chisel assertions required')
    rows,cols=caps['rows'],caps['cols'];period=caps['feedback_issue_interval'];hop=caps['hop_latency']
    payload=' '.join(f'{v:x}'for v in [1]+[0]*(cols-1)+[1]+[0]*(rows-1)+[1]+[0]*(cols-1))
    first=f'I 4 0 1 0 1 {payload}\n';second=f'I 4 0 0 1 1 {payload}\n'
    legal=build/'hazard_legal.txt';legal.write_text(first+f'W {period-1}\n'+second+'D\nR 0 0 0 2\n')
    result=subprocess.run([str(build/'sim'),str(legal)],check=True,capture_output=True,text=True,timeout=120)
    line=next(line for line in result.stdout.splitlines()if line.startswith('R '));_,tag,data,flags,cycles,macs,commits=line.split()
    drain=max((rows+cols-1)*hop+1,(cols-1)*hop+caps['stages']+3+caps['pe_commit_latency']+
        caps['pe_state_read_latency']+caps['counter_settle_cycles']-2)
    predicted=period+drain+caps['array_read_latency']+caps.get('array_busy_release_latency',0)
    if(int(tag),int(data,16),int(flags),int(macs),int(cycles))!=(2,2,0,2,predicted):
        raise AssertionError('legal dependent waves differ from value/cycle model')
    bad=build/'hazard_invalid.txt';bad.write_text(first+second+'D\n')
    def no_core_dump():resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    try:invalid=subprocess.run([str(build/'sim'),str(bad)],capture_output=True,text=True,timeout=120,preexec_fn=no_core_dump)
    except subprocess.TimeoutExpired as error:
        output=error.stdout or b'';output=output.decode(errors='replace')if isinstance(output,bytes)else output
        (build/'hazard_invalid.log').write_text(output)
        (build/'verification_hazard_host_failed.json').write_text(json.dumps(dict(passed=False,
            scope='scheduled_mesh_context_reuse',status='host_timeout_after_assertion'if 'Assertion failed'in output else'host_timeout',
            rtl_assertion_observed='streaming wave reused an unfinished context or active seed'in output,
            host_termination_failed=True,rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256']),indent=2)+'\n')
        raise
    log=build/'hazard_invalid.log';log.write_text(invalid.stdout+invalid.stderr)
    if invalid.returncode==0 or 'streaming wave reused an unfinished context or active seed'not in log.read_text():
        raise AssertionError('early context reuse was not rejected by the mesh protocol assertion')
    report=dict(passed=True,scope='scheduled_mesh_context_reuse',legal_waves=2,
        legal_cycles=int(cycles),cycle_model_prediction=predicted,scheduled_feedback_interval=period,
        legal_issue_gap_cycles=period,invalid_early_reuse_asserted=True,host_assertion_termination_checked=True,
        controller_validation='separate descriptor integration regression',
        protocol='valid-only mesh; hardware TileSystem cooldown schedules dependencies',
        stages=caps['stages'],rows=rows,cols=cols,rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'],
        legal_trace_sha256=hashlib.sha256(legal.read_bytes()).hexdigest(),invalid_trace_sha256=hashlib.sha256(bad.read_bytes()).hexdigest())
    (build/'verification_hazard.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build-dir',type=Path,required=True)
    check(parser.parse_args().build_dir.resolve())
