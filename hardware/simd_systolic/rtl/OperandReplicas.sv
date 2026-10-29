// SPDX-License-Identifier: Apache-2.0
// Narrow floating A is a scalar broadcast in the matrix mesh. Replicate its
// packed input at the existing first preparation boundary: one local source
// register drives each expansion slice, rather than one bit driving sixteen
// distributed subnormal/format decoders. FP64 retains its ordinary 64-bit path.
(* keep_hierarchy=1 *)
module SimdPackedARegister(input logic clock,input logic [31:0] data,
 output logic [31:0] value);
 always_ff @(posedge clock)value<=data;
endmodule
module SimdPackedAReplicas(input logic clock,input logic [31:0] data,
 output logic [511:0] value);
 for(genvar lane=0;lane<16;lane++)begin: local_source
  (* keep=1,dont_touch=1,keep_hierarchy=1 *)
  SimdPackedARegister replica(.clock(clock),.data(data),.value(value[32*lane+:32]));
 end
endmodule
