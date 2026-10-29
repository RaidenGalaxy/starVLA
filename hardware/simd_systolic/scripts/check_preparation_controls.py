#!/usr/bin/env python3
"""Prove local decoded holds equal packed format/op holds at the same edge."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def check(out):
    out.mkdir(parents=True,exist_ok=True)
    source=out/'ControlReplicas.sv';shutil.copy2(ROOT/'rtl/ControlReplicas.sv',source)
    miter=out/'miter.sv'
    miter.write_text('''module PreparationMiter(input clock,capture,
 input [3:0] fmt,ctx,input [2:0] op,input [15:0] pending,previous_pending,
 input seed,previous_seed,output equivalent);
 reg [3:0] old_fmt;
 reg [2:0] old_op;
 always @(posedge clock)if(capture)begin old_fmt<=fmt;old_op<=op;end
 wire [3:0] selected_fmt=capture?fmt:old_fmt;
 wire [2:0] selected_op=capture?op:old_op;
 wire [303:0] old_controls,local_controls;
 SimdPreparationControlReplicas original(clock,selected_fmt,selected_op,old_controls);
 SimdElasticPreparationControlReplicas updated(clock,capture,fmt,op,local_controls);
 wire old_available=!pending[ctx]&&!previous_pending[ctx]&&!seed&&!previous_seed;
 wire [15:0] unavailable=pending|previous_pending;
 wire new_available=!unavailable[ctx]&&!seed&&!previous_seed;
 assign equivalent=(old_controls==local_controls)&&(old_available==new_available);
endmodule
''')
    script=out/'proof.ys'
    script.write_text(f'''read_verilog -sv {source} {miter}
hierarchy -check -top PreparationMiter
proc
setattr -mod -unset keep_hierarchy
setattr -unset keep_hierarchy
flatten
opt_clean
sat -seq 5 -set-def-inputs -set-at 1 capture 1 -prove equivalent 1 -prove-skip 1 -verify
''')
    log=out/'proof.log'
    with log.open('w')as stream:
        subprocess.run(['yosys','-Q','-T','-s',str(script)],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=120)
    if 'SAT proof finished - no model found: SUCCESS!'not in log.read_text():raise AssertionError('missing successful proof')
    report=dict(passed=True,scope='elastic_preparation_control_equivalence',local_words=16,
        format_encodings=16,operation_encodings=8,arbitrary_hold_capture_sequences=True,
        initial_state_unconstrained=True,first_clock_capture_required=True,comparison_after_first_capture=True,
        unchanged_preparation_latency=True,dependency_bitmap_selection_equivalent=True,
        arbitrary_pending_and_previous_pending=True,timing_validated=False,
        limitation='Proves control transformation; full PE arithmetic, queue, reset and extracted timing are separate checks.')
    for name,path in [('source',source),('miter',miter),('proof_script',script),('proof_log',log),('checker',Path(__file__))]:
        report[name+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'.build/preparation_control_proof')
    check(parser.parse_args().out.resolve())
