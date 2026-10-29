// SPDX-License-Identifier: Apache-2.0
// Keep local register instances distinct through logic synthesis. Each copy
// drives one 32-bit result slice instead of a single global format selector.
(* keep_hierarchy=1 *)
module SimdResetSynchronizer(input logic clock,externalReset,output logic value);
 (* ASYNC_REG="TRUE",keep=1 *) logic first,second;
 always_ff @(posedge clock or posedge externalReset)
  if(externalReset)begin first<=1'b1;second<=1'b1;end
  else begin first<=1'b0;second<=first;end
 assign value=second;
endmodule
(* keep_hierarchy=1 *)
module SimdToggleSynchronizer(input logic clock,reset,data,output logic value);
 (* ASYNC_REG="TRUE",keep=1 *) logic first,second;
 always_ff @(posedge clock or posedge reset)
  if(reset)begin first<=1'b0;second<=1'b0;end
  else begin first<=data;second<=first;end
 assign value=second;
endmodule
(* keep_hierarchy=1 *)
module SimdContextReadRegister(input logic clock,input logic [3:0] ctx,
 input logic [15:0] initialized,input logic clear,output logic [15:0] value);
 logic [3:0] address;
 logic [15:0] initialized_snapshot;
 logic cleared;
 always_ff @(posedge clock)begin
  address<=ctx;initialized_snapshot<=initialized;cleared<=clear;
 end
 for(genvar c=0;c<16;c++)begin: context_decode
  assign value[c]=!cleared&&initialized_snapshot[c]&&(address==c);
 end
endmodule
module SimdContextReadReplicas(input logic clock,input logic [3:0] ctx,
 input logic [15:0] initialized,input logic clear,output logic [255:0] value);
 // Capture the initialized bitmap at the original request boundary. Selecting
 // initialized[ctx] before that register was the measured 2x2 critical path:
 // a distant wave context drove a 16:1 mux and every word's enable register.
 // Local selection now shares the existing second preparation stage with
 // state reads. Snapshotting (rather than using live initialized bits) keeps
 // simultaneous first commits/seeds from changing the old read semantics.
 for(genvar word=0;word<16;word++)begin: local_word
  (* keep=1,dont_touch=1,keep_hierarchy=1 *)
  SimdContextReadRegister replica(.clock(clock),.ctx(ctx),.initialized(initialized),.clear(clear),.value(value[16*word+:16]));
 end
endmodule
// Combinational copies retain the live initialized/commit state at the
// original read/write clock. Moving this decision before a register would
// change observation of a simultaneous first commit or state seed.
(* keep_hierarchy=1 *)
module SimdObservationSelect(input logic [15:0] address,initialized,
 output logic [15:0] selected);
 assign selected=address & initialized;
endmodule
// Each 32-bit observation slice maintains the same live initialized bitmap
// from its existing local commit/seed decoders. No extra read clock is added.
// The previous shared bitmap drove all sixteen far-apart observation muxes.
(* keep_hierarchy=1 *)
module SimdObservationState(input logic clock,reset,input logic [15:0] address,
 input logic [15:0] commitAddress,input logic commit,input logic [15:0] seedAddress,
 output logic [15:0] selected);
 logic [15:0] initialized;
 always_ff @(posedge clock)
  if(reset)initialized<=16'b0;
  else initialized<=initialized | (commitAddress & {16{commit}}) | seedAddress;
 assign selected=address & initialized;
endmodule
(* keep_hierarchy=1 *)
module SimdStateWriteSelect(input logic [15:0] address,input logic commit,
 output logic [15:0] write);
 assign write=address & {16{commit}};
endmodule
// Seed data and per-word one-hot enables share a registered boundary. The
// command's context bits no longer drive the complete accumulator mux bank.
(* keep_hierarchy=1 *)
module SimdPeSeedWord(input logic clock,reset,input logic [3:0] ctx,
 input logic valid,input logic [31:0] data,output logic [31:0] value,
 output logic [15:0] selected);
 always_ff @(posedge clock)value<=data;
 for(genvar c=0;c<16;c++)begin: local_seed_decode
  always_ff @(posedge clock)if(reset)selected[c]<=1'b0;else selected[c]<=valid&&(ctx==c);
 end
endmodule
// Fixed-latency commit words: local validity and context copies accompany
// unconditional data registers. Reserved response FIFOs handle backpressure
// independently of these state writes.
(* keep_hierarchy=1 *)
module SimdPeCommitWord(input logic clock,reset,ready,valid,input logic [31:0] data,input logic [3:0] ctx,
 output logic [31:0] value,output logic [15:0] address,output logic validOut);
 always_ff @(posedge clock) begin
  value<=data;
  if(reset)validOut<=1'b0;else validOut<=valid;
 end
 for(genvar c=0;c<16;c++)always_ff @(posedge clock)address[c]<=ctx==c;
endmodule
(* keep_hierarchy=1 *)
module SimdPeElasticCommitWord(input logic clock,reset,ready,valid,input logic [31:0] data,input logic [3:0] ctx,
 output logic [31:0] value,output logic [15:0] address,output logic validOut);
 wire advance=!validOut||ready;
 always_ff @(posedge clock)begin
  if(advance)value<=data;
  if(reset)validOut<=1'b0;else if(advance)validOut<=valid;
 end
 for(genvar c=0;c<16;c++)always_ff @(posedge clock)if(advance)address[c]<=ctx==c;
endmodule
(* keep_hierarchy=1 *)
module SimdControlRegister(
 input logic clock, input logic [28:0] data,
 output logic [28:0] value);
 always_ff @(posedge clock) value<=data;
endmodule
// Local pointer/state per 32-bit FIFO slice avoids a global address driving
// a 570-bit read mux. Width and capacity are fixed at the Slang boundary.
(* keep_hierarchy=1 *)
module SimdResponseBank #(parameter integer DEPTH=14)(
 input logic clock,reset,write,ready,input logic [31:0] data,output logic [31:0] value);
 logic [3:0] write_pointer,read_pointer;
 logic [31:0] storage[DEPTH];
 logic [$clog2(DEPTH+1)-1:0] count;
 logic [31:0] head;
 // Each word owns the queue occupancy. A shared top-level count must not
 // feed every head-register enable through a global dequeue decoder.
 wire read=ready&&(count!=0);
 wire [3:0] next_read=read_pointer==DEPTH-1?4'd0:read_pointer+4'd1;
 always_ff @(posedge clock) begin
  if(reset) begin write_pointer<=0;read_pointer<=0;count<=0;end
  else begin
   if(write&&!read)count<=count+1'b1;
   if(read&&!write)count<=count-1'b1;
   // Prefetch the next head on consumption. The empty and one-entry cases
   // take incoming data directly, preserving the one-clock enqueue latency
   // even when the last entry is consumed and replaced on the same edge.
   if(write&&(count==0||(read&&count==1)))head<=data;
   else if(read&&count>1)head<=storage[next_read];
   if(write) begin storage[write_pointer]<=data;write_pointer<=write_pointer==DEPTH-1?0:write_pointer+1'b1;end
   if(read) read_pointer<=next_read;
  end
 end
 assign value=head;
endmodule
module SimdResponseBank14(input logic clock,reset,write,ready,input logic [31:0] data,output logic [31:0] value);
 (* keep=1,dont_touch=1,keep_hierarchy=1 *)
 SimdResponseBank #(.DEPTH(14)) bank(.*);
endmodule
module SimdResponseBank10(input logic clock,reset,write,ready,input logic [31:0] data,output logic [31:0] value);
 (* keep=1,dont_touch=1,keep_hierarchy=1 *)
 SimdResponseBank #(.DEPTH(10)) bank(.*);
endmodule
module SimdResponseBank15(input logic clock,reset,write,ready,input logic [31:0] data,output logic [31:0] value);
 (* keep=1,dont_touch=1,keep_hierarchy=1 *)
 SimdResponseBank #(.DEPTH(15)) bank(.*);
endmodule
module SimdResponseBank11(input logic clock,reset,write,ready,input logic [31:0] data,output logic [31:0] value);
 (* keep=1,dont_touch=1,keep_hierarchy=1 *)
 SimdResponseBank #(.DEPTH(11)) bank(.*);
endmodule
(* keep_hierarchy=1 *)
module SimdValidRegister(input logic clock,reset,data,output logic value);
 always_ff @(posedge clock)if(reset)value<=1'b0;else value<=data;
endmodule
module SimdValidReplicas(input logic clock,reset,data,output logic [15:0] value);
 for(genvar i=0;i<16;i++)begin: local_valid
  (* keep=1,dont_touch=1,keep_hierarchy=1 *)
  SimdValidRegister replica(.clock(clock),.reset(reset),.data(data),.value(value[i]));
 end
endmodule
(* keep_hierarchy=1 *)
module SimdBypassControlRegister(input logic clock,reset,data,output logic value);
 always_ff @(posedge clock) if(reset) value<=1'b1;else value<=data;
endmodule
module SimdBypassControlReplicas(input logic clock,reset,data,output logic [15:0] value);
 for(genvar i=0;i<16;i++) begin: local_control
   (* keep=1, dont_touch=1, keep_hierarchy=1 *)
   SimdBypassControlRegister replica(.clock(clock),.reset(reset),.data(data),.value(value[i]));
 end
endmodule
module SimdControlReplicas(
 input logic clock,input logic [28:0] data,
 output logic [463:0] value);
 for(genvar i=0;i<16;i++) begin: local_control
   (* keep=1, dont_touch=1, keep_hierarchy=1 *)
   SimdControlRegister replica(.clock(clock),.data(data),.value(value[i*29+:29]));
 end
endmodule
(* keep_hierarchy=1 *)
module SimdReadControlRegister(input logic clock,input logic [1:0] data,output logic [1:0] value);
 always_ff @(posedge clock) value<=data;
endmodule
module SimdReadControlReplicas(input logic clock,input logic [1:0] data,output logic [31:0] value);
 for(genvar i=0;i<16;i++) begin: local_control
   (* keep=1, dont_touch=1, keep_hierarchy=1 *)
   SimdReadControlRegister replica(.clock(clock),.data(data),.value(value[i*2+:2]));
 end
endmodule
(* keep_hierarchy=1 *)
module SimdOpControlRegister(input logic clock,input logic [2:0] data,output logic [2:0] value);
 always_ff @(posedge clock) value<=data;
endmodule
(* keep_hierarchy=1 *)
module SimdSourceFormatDecode(input logic [3:0] fmt,output logic [10:0] value);
 for(genvar f=0;f<11;f++) assign value[f]=(fmt==f);
endmodule
module SimdSourceFormatReplicas(input logic [3:0] fmt,output logic [175:0] value);
 for(genvar i=0;i<16;i++) begin: local_control
   (* keep=1, dont_touch=1, keep_hierarchy=1 *)
   SimdSourceFormatDecode replica(.fmt(fmt),.value(value[11*i+:11]));
 end
endmodule
(* keep_hierarchy=1 *)
module SimdSourceOpDecode(input logic [2:0] op,output logic [7:0] value);
 for(genvar f=0;f<8;f++) assign value[f]=(op==f);
endmodule
module SimdSourceOpReplicas(input logic [2:0] op,output logic [127:0] value);
 for(genvar i=0;i<16;i++) begin: local_control
   (* keep=1,dont_touch=1,keep_hierarchy=1 *)
   SimdSourceOpDecode replica(.op(op),.value(value[8*i+:8]));
 end
endmodule
(* keep_hierarchy=1 *)
module SimdPreparationControlRegister(input logic clock,input logic [3:0] fmt,input logic [2:0] op,
 output logic [18:0] value);
 for(genvar f=0;f<11;f++)always_ff @(posedge clock)value[f]<=fmt==f;
 for(genvar o=0;o<8;o++)always_ff @(posedge clock)value[11+o]<=op==o;
endmodule
module SimdPreparationControlReplicas(input logic clock,input logic [3:0] fmt,input logic [2:0] op,
 output logic [303:0] value);
 for(genvar i=0;i<16;i++)begin: local_preparation
  (* keep=1,dont_touch=1,keep_hierarchy=1 *)
  SimdPreparationControlRegister replica(.clock(clock),.fmt(fmt),.op(op),.value(value[19*i+:19]));
 end
endmodule
// Decode before the local hold mux. Selecting a packed held/new format
// before decode put the registered dependency check on a measured long
// format-control path in the elastic standalone PE.
(* keep_hierarchy=1 *)
module SimdElasticPreparationControlRegister(input logic clock,enable,
 input logic [3:0] fmt,input logic [2:0] op,output logic [18:0] value);
 for(genvar f=0;f<11;f++)always_ff @(posedge clock)if(enable)value[f]<=fmt==f;
 for(genvar o=0;o<8;o++)always_ff @(posedge clock)if(enable)value[11+o]<=op==o;
endmodule
module SimdElasticPreparationControlReplicas(input logic clock,enable,
 input logic [3:0] fmt,input logic [2:0] op,output logic [303:0] value);
 for(genvar i=0;i<16;i++)begin: local_preparation
  (* keep=1,dont_touch=1,keep_hierarchy=1 *)
  SimdElasticPreparationControlRegister replica(.clock(clock),.enable(enable),.fmt(fmt),.op(op),.value(value[19*i+:19]));
 end
endmodule
// Own the buffered dependency state beside each operand/control word. A
// central held-context mux must not drive every operand register enable.
// All copies see the same accepted request and busy bitmap, and retain the
// original preparation boundary and initialization snapshot clock.
(* keep_hierarchy=1 *)
module SimdElasticPreparationWord(input logic clock,reset,accepted,blocked,
 input logic [15:0] unavailable,initialized,input logic [3:0] ctx,fmt,
 input logic [2:0] op,input logic clear,input logic [31:0] data,
 output logic [31:0] value,output logic [18:0] controls,
 output logic [15:0] selected,output logic capture,output logic [5:0] state);
 logic occupied,cleared;
 logic [3:0] context_id;
 logic [15:0] initialized_snapshot;
 assign capture=!occupied||(!unavailable[context_id]&&!blocked);
 assign state={occupied,context_id,cleared};
 always_ff @(posedge clock)begin
  if(reset)occupied<=1'b0;else if(capture)occupied<=accepted;
  initialized_snapshot<=initialized;
  if(capture)begin
   value<=data;context_id<=ctx;cleared<=clear;
   for(integer f=0;f<11;f++)controls[f]<=fmt==f;
   for(integer o=0;o<8;o++)controls[11+o]<=op==o;
  end
 end
 for(genvar c=0;c<16;c++)begin: local_context_decode
  assign selected[c]=!cleared&&initialized_snapshot[c]&&(context_id==c);
 end
endmodule
// S8 fine normalization controls are registered beside 16-bit data segments.
// Preserve CE semantics for scalar/gate verification as well as the array.
(* keep_hierarchy=1 *)
module SimdShiftControlRegister(input logic clock,ce,input logic [3:0] data,output logic [3:0] value);
 always_ff @(posedge clock) if(ce) value<=data;
endmodule
module SimdShiftControlReplicas(input logic clock,ce,input logic [3:0] data,output logic [63:0] value);
 for(genvar i=0;i<16;i++) begin: local_control
   (* keep=1,dont_touch=1,keep_hierarchy=1 *)
   SimdShiftControlRegister replica(.clock(clock),.ce(ce),.data(data),.value(value[4*i+:4]));
 end
endmodule
// Preserve complete local wavefront banks across logic synthesis. Metadata
// with the same value on one diagonal must not become one global FF driver.
(* keep_hierarchy=1 *)
module SimdWaveRegister(input logic clock,reset,valid,input logic [120:0] data,
 output logic [120:0] value,output logic validOut);
 always_ff @(posedge clock) begin
  value<=data;
  if(reset)validOut<=1'b0;else validOut<=valid;
 end
endmodule
