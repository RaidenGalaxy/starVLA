#!/usr/bin/env python3
"""Prove the retimed context selection against the previous clock boundary."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def check(out):
    out.mkdir(parents=True,exist_ok=True)
    source=out/'ControlReplicas.sv';shutil.copy2(ROOT/'rtl/ControlReplicas.sv',source)
    miter=out/'miter.sv'
    miter.write_text('''module ContextReadMiter(input clock,input [3:0] ctx,
 input [15:0] initialized,input clear,output equivalent);
 wire [255:0] actual;
 SimdContextReadReplicas dut(clock,ctx,initialized,clear,actual);
 // Previous implementation selected initialized[ctx] before the register.
 reg [3:0] old_address;
 reg old_active;
 always @(posedge clock) begin
   old_address<=ctx;old_active<=initialized[ctx]&&!clear;
 end
 wire [15:0] expected=old_active ? (16'b1<<old_address) : 16'b0;
 assign equivalent=actual=={16{expected}};
endmodule
''')
    script=out/'prove.ys'
    script.write_text(f'''read_verilog -sv {source} {miter}
hierarchy -check -top ContextReadMiter
proc
setattr -mod -unset keep_hierarchy
setattr -unset keep_hierarchy
flatten
opt_clean
sat -seq 2 -set-def-inputs -prove equivalent 1 -prove-skip 1 -verify
''')
    with(out/'proof.log').open('w')as log:
        subprocess.run(['yosys','-Q','-T','-s',str(script)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    if 'SAT proof finished - no model found: SUCCESS!'not in(out/'proof.log').read_text():
        raise AssertionError('no successful sequential equivalence proof')
    result=dict(passed=True,scope='context_selection_sequential_equivalence',contexts=16,local_words=16,
                arbitrary_initialized_bitmap=True,arbitrary_ctx_and_clear_each_clock=True,
                initial_state_unconstrained=True,comparison_after_first_capture=True,
                unchanged_preparation_latency=True,timing_certification=False,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                miter_sha256=hashlib.sha256(miter.read_bytes()).hexdigest(),
                proof_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                proof_log_sha256=hashlib.sha256((out/'proof.log').read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'.build/context_read_proof')
    check(p.parse_args().out.resolve())
