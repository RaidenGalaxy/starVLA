#!/usr/bin/env python3
"""Prove local packed-A captures equal the old broadcast at its clock boundary."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def check(out):
    out.mkdir(parents=True,exist_ok=True)
    source=out/'OperandReplicas.sv';shutil.copy2(ROOT/'rtl/OperandReplicas.sv',source)
    miter=out/'miter.sv'
    miter.write_text('''module PackedAMiter(input clock,input [63:0] operand,
 output equivalent);
 wire [511:0] actual;
 SimdPackedAReplicas dut(clock,operand[31:0],actual);
 // The old first preparation register captured the complete request.
 reg [63:0] previous_request_a;
 always @(posedge clock) previous_request_a<=operand;
 assign equivalent=actual=={16{previous_request_a[31:0]}};
endmodule
''')
    script=out/'prove.ys'
    script.write_text(f'''read_verilog -sv {source} {miter}
hierarchy -check -top PackedAMiter
proc
setattr -mod -unset keep_hierarchy
setattr -unset keep_hierarchy
flatten
opt_clean
sat -seq 3 -set-def-inputs -prove equivalent 1 -prove-skip 1 -verify
''')
    with(out/'proof.log').open('w')as log:
        subprocess.run(['yosys','-Q','-T','-s',str(script)],check=True,
                       stdout=log,stderr=subprocess.STDOUT,timeout=120)
    if 'SAT proof finished - no model found: SUCCESS!'not in(out/'proof.log').read_text():
        raise AssertionError('no successful sequential equivalence proof')
    result=dict(passed=True,scope='packed_a_broadcast_sequential_equivalence',
                lanes=16,local_register_bits=32,arbitrary_input_each_clock=True,
                initial_state_unconstrained=True,comparison_after_first_capture=True,
                unchanged_preparation_latency=True,arithmetic_equivalence=False,
                timing_certification=False)
    for name,path in [('source',source),('miter',miter),('proof_script',script),('proof_log',out/'proof.log')]:
        result[name+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'.build/packed_a_proof')
    check(p.parse_args().out.resolve())
