#!/usr/bin/env python3
"""Prove distributed capture against the previous central request boundary."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def check(out):
    out.mkdir(parents=True, exist_ok=True)
    source = out / 'ControlReplicas.sv'
    shutil.copy2(ROOT / 'rtl/ControlReplicas.sv', source)
    miter = out / 'miter.sv'
    miter.write_text('''module CaptureMiter(input clock,reset,accepted,blocked,
 input [15:0] unavailable,initialized,input [3:0] ctx,fmt,
 input [2:0] op,input clear,input [31:0] data,output equivalent);
 reg occupied,held_clear,selected_clear;
 reg [3:0] held_ctx,held_fmt,selected_ctx;
 reg [2:0] held_op;
 reg [31:0] held_data;
 reg [15:0] snapshot;
 reg [18:0] old_controls;
 wire capture=!occupied||(!unavailable[held_ctx]&&!blocked);
 wire [3:0] next_fmt=capture?fmt:held_fmt;
 wire [2:0] next_op=capture?op:held_op;
 wire [15:0] old_selected;
 always @(posedge clock)begin
  if(reset)occupied<=0;else if(capture)occupied<=accepted;
  if(capture)begin
   held_ctx<=ctx;held_clear<=clear;held_fmt<=fmt;held_op<=op;held_data<=data;
  end
  snapshot<=initialized;
  selected_ctx<=capture?ctx:held_ctx;
  selected_clear<=capture?clear:held_clear;
  for(integer f=0;f<11;f=f+1)old_controls[f]<=next_fmt==f;
  for(integer o=0;o<8;o=o+1)old_controls[11+o]<=next_op==o;
 end
 for(genvar c=0;c<16;c=c+1)
  assign old_selected[c]=!selected_clear&&snapshot[c]&&(selected_ctx==c);
 wire [31:0] value;
 wire [18:0] controls;
 wire [15:0] selected;
 wire local_capture;
 wire [5:0] state;
 SimdElasticPreparationWord updated(.clock(clock),.reset(reset),.accepted(accepted),
  .blocked(blocked),.unavailable(unavailable),.initialized(initialized),
  .ctx(ctx),.fmt(fmt),.op(op),.clear(clear),.data(data),.value(value),
  .controls(controls),.selected(selected),.capture(local_capture),.state(state));
 assign equivalent=(capture==local_capture)&&(state=={occupied,held_ctx,held_clear})
  &&(value==held_data)&&(controls==old_controls)&&(selected==old_selected);
endmodule
''')
    prelude = f'''read_verilog -sv {source} {miter}
hierarchy -check -top CaptureMiter
proc
setattr -mod -unset keep_hierarchy
setattr -unset keep_hierarchy
flatten
opt_clean
'''
    commands = {
        'bounded': 'sat -seq 10 -set-def-inputs -set-at 1 reset 1 -prove equivalent 1 -prove-skip 2 -verify\n',
        'induction': 'sat -seq 3 -tempinduct-inductonly -maxsteps 10 -set-def-inputs -prove equivalent 1 -verify\n',
    }
    for name, command in commands.items():
        script = out / (name + '.ys')
        script.write_text(prelude + command)
        log = out / (name + '.log')
        with log.open('w') as stream:
            subprocess.run(['yosys', '-Q', '-T', '-s', str(script)],
                           stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=180)
        if 'SUCCESS!' not in log.read_text():
            raise AssertionError('missing proof success: ' + name)
    result = dict(passed=True, scope='distributed_request_capture_equivalence',
                  local_words=16, data_bits_per_word=32,
                  format_encodings=16, operation_encodings=8,
                  bounded_clocks=10, temporal_induction=True,
                  arbitrary_busy_seed_reset_and_capture_sequences=True,
                  initial_register_state_unconstrained=True,
                  first_clock_reset_required=True,
                  comparison_after_reset_and_first_capture=True,
                  same_preparation_boundary=True, timing_validated=False,
                  limitation='Local controller, held data, decode and initialization snapshot only; full PE and physical checks are separate.')
    files = {'source': source, 'miter': miter, 'checker': Path(__file__),
             'pe_implementation': ROOT / 'src/main/scala/simd/Array.scala'}
    files.update({name + '_' + kind: out / (name + '.' + ext)
                  for name in commands for kind, ext in [('script', 'ys'), ('log', 'log')]})
    result.update({name + '_sha256': hashlib.sha256(path.read_bytes()).hexdigest()
                   for name, path in files.items()})
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / '.build/elastic_preparation_word_proof')
    check(parser.parse_args().out.resolve())
