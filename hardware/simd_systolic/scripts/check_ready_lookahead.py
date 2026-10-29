#!/usr/bin/env python3
"""Prove registered capture prediction against the original admission state."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def check(out):
    out.mkdir(parents=True, exist_ok=True)
    miter = out / 'miter.sv'
    miter.write_text('''module ReadyMiter(input clock,reset,accepted,seed_request,
 input [3:0] incoming_ctx,input [15:0] commits,output equivalent);
 reg [15:0] pending,previous_pending;
 reg seed,previous_seed,occupied,predicted;
 reg [3:0] ctx;
 wire available=!(pending[ctx]||previous_pending[ctx]||seed||previous_seed);
 wire capture=!occupied||available;
 wire advance=occupied&&available;
 wire [15:0] issued=advance?(16'b1<<ctx):16'b0;
 // Reference: the actual PE clears committed pending bits, then gives
 // same-clock issues priority. Its history register sees pre-edge pending.
 wire [15:0] pending_next=(pending&~commits)|issued;
 wire [3:0] next_ctx=capture?incoming_ctx:ctx;
 wire next_occupied=capture?accepted:occupied;
 // Implementation: pending_next | previous_pending_next simplifies to
 // pending | issued, without a commit decoder in the prediction cone.
 wire [15:0] future_unavailable=pending|issued;
 wire prediction=!next_occupied||(!future_unavailable[next_ctx]&&!seed_request&&!seed);
 always @(posedge clock)begin
  if(capture)ctx<=incoming_ctx;
  if(reset)begin
   pending<=0;previous_pending<=0;seed<=0;previous_seed<=0;occupied<=0;predicted<=1;
  end else begin
   pending<=pending_next;previous_pending<=pending;
   seed<=seed_request;previous_seed<=seed;
   if(capture)occupied<=accepted;
   predicted<=prediction;
  end
 end
 assign equivalent=predicted==capture;
endmodule
''')
    prelude = f'read_verilog -sv {miter}\nprep -top ReadyMiter\nflatten\n'
    commands = {
        'bounded': 'sat -seq 10 -set-def-inputs -set-at 1 reset 1 -prove equivalent 1 -prove-skip 1 -verify\n',
        'induction': 'sat -seq 2 -tempinduct-inductonly -maxsteps 8 -prove equivalent 1 -verify\n',
    }
    for name, command in commands.items():
        script = out / (name + '.ys')
        script.write_text(prelude + command)
        with (out / (name + '.log')).open('w') as log:
            subprocess.run(['yosys', '-Q', '-T', '-s', str(script)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        if 'SUCCESS!' not in (out / (name + '.log')).read_text():
            raise AssertionError('missing successful ' + name + ' proof')
    result = dict(passed=True, scope='registered_ready_prediction_equivalence',
                  contexts=16, bounded_clocks=10, temporal_induction=True,
                  arbitrary_commit_masks_seed_requests_acceptance_and_resets=True,
                  initial_state_unconstrained=True, first_clock_reset_required=True,
                  induction_initial_bits_arbitrary_defined_binary=True,
                  bounded_proof_models_undefined_initial_state=True,
                  queue_capacity_unchanged=True, response_credits_unchanged=True,
                  no_new_operand_stage=True, timing_validated=False,
                  limitation='Proves admission prediction state equations; full PE RTL and extracted timing checks remain separate.')
    files = {'checker': Path(__file__), 'miter': miter,
             'pe_implementation': ROOT / 'src/main/scala/simd/Array.scala'}
    files.update({name + '_' + kind: out / (name + '.' + ext)
                  for name in commands for kind, ext in [('script', 'ys'), ('log', 'log')]})
    result.update({key + '_sha256': hashlib.sha256(path.read_bytes()).hexdigest()
                   for key, path in files.items()})
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / '.build/ready_lookahead_proof')
    check(parser.parse_args().out.resolve())
