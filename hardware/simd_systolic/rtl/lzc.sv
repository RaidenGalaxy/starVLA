// SPDX-License-Identifier: Apache-2.0
// Balanced leading/trailing zero tree; independent implementation.
module lzc #(parameter int unsigned WIDTH=2, parameter bit MODE=0,
 parameter int unsigned CNT_WIDTH=(WIDTH>1 ? $clog2(WIDTH) : 1))
 (input logic [WIDTH-1:0] in_i, output logic [CNT_WIDTH-1:0] cnt_o,
 output logic empty_o);
 localparam int N=2**$clog2(WIDTH);
 wire [2*N-1:1] present;
 wire [CNT_WIDTH-1:0] idx[2*N-1:1];
 for(genvar i=0;i<N;i++) begin
   if(i<WIDTH) assign present[N+i]=MODE ? in_i[WIDTH-1-i] : in_i[i];
   else assign present[N+i]=1'b0;
   assign idx[N+i]=CNT_WIDTH'(i);
 end
 for(genvar i=1;i<N;i++) begin
   assign present[i]=present[2*i]|present[2*i+1];
   assign idx[i]=present[2*i] ? idx[2*i] : idx[2*i+1];
 end
 assign empty_o=~present[1];
 assign cnt_o=present[1] ? idx[1] : CNT_WIDTH'(WIDTH-1);
endmodule
