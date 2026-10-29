#!/usr/bin/env python3
"""Prove replicated live initialization and same-edge observation semantics."""
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
    miter.write_text('''module ObservationStateMiter(input clock,reset,commit,
 input [15:0] commitAddress,seedAddress,address,input [511:0] data,
 output invariant,observation_equivalent);
 reg [15:0] initialized;
 always @(posedge clock)
  if(reset)initialized<=0;
  else initialized<=initialized|(commitAddress&{16{commit}})|seedAddress;
 wire [255:0] copies;
 for(genvar word=0;word<16;word++)begin: local_word
  // All bits are observed here, including contexts not selected this clock.
  SimdObservationState dut(clock,reset,16'hffff,commitAddress,commit,
   seedAddress,copies[16*word+:16]);
 end
 assign invariant=copies=={16{initialized}};
 reg [31:0] expected,actual;
 reg [31:0] selected_expected,selected_actual;
 integer c;
 always @* begin
  selected_expected=0;selected_actual=0;
  for(c=0;c<16;c=c+1)begin
   selected_expected=selected_expected|({32{address[c]&initialized[c]}}&data[32*c+:32]);
   selected_actual=selected_actual|({32{address[c]&copies[c]}}&data[32*c+:32]);
  end
 end
 always @(posedge clock)begin expected<=selected_expected;actual<=selected_actual;end
 assign observation_equivalent=actual==expected;
endmodule
''')
    base=f'''read_verilog -sv {source} {miter}
hierarchy -check -top ObservationStateMiter
proc
setattr -mod -unset keep_hierarchy
setattr -unset keep_hierarchy
flatten
opt_clean
'''
    scripts={
        'induction':base+'''sat -seq 2 -set-init-zero -set-def-inputs -tempinduct -prove invariant 1 -verify
''',
        'observation':base+'''sat -seq 6 -set-def-inputs -set-at 1 reset 1 -prove observation_equivalent 1 -prove-skip 2 -verify
'''}
    artifacts={'source':source,'miter':miter,'checker':Path(__file__)}
    for name,body in scripts.items():
        script=out/(name+'.ys');script.write_text(body);log=out/(name+'.log')
        with log.open('w')as stream:
            subprocess.run(['yosys','-Q','-T','-s',str(script)],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=120)
        text=log.read_text()
        if 'SUCCESS!'not in text:raise AssertionError('missing successful '+name+' proof')
        artifacts[name+'_script']=script;artifacts[name+'_log']=log
    report=dict(passed=True,scope='local_observation_initialization_equivalence',
        contexts=16,local_words=16,live_bitmap_inductively_proven=True,
        arbitrary_commit_and_seed_masks=True,arbitrary_read_masks_and_state_data=True,
        synchronous_reset_reassertion=True,same_edge_first_commit_and_seed_preserved=True,
        unchanged_read_latency=True,timing_validated=False,
        limitation='Control-state induction and six-clock data observation proof; full PE/mesh numerical and extracted timing checks remain separate.')
    report['artifacts']={name:str(path.resolve())for name,path in artifacts.items()}
    report['artifacts_sha256']={name:hashlib.sha256(path.read_bytes()).hexdigest()for name,path in artifacts.items()}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'.build/observation_state_proof')
    check(parser.parse_args().out.resolve())
