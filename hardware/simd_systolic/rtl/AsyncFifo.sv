// SPDX-License-Identifier: Apache-2.0
// Dual-clock FIFO: Gray pointers, two synchronizer flops per crossing.
module AsyncFifo #(parameter int WIDTH=64, parameter int ADDR=4)(
 input logic wclk,wrst,rclk,rrst,
 input logic wvalid,output logic wready,input logic [WIDTH-1:0] wdata,
 output logic rvalid,input logic rready,output logic [WIDTH-1:0] rdata);
 localparam int P=ADDR+1;
 logic [WIDTH-1:0] mem[0:(1<<ADDR)-1];
 logic [P-1:0] wb,rb,wg,rg;
 (* ASYNC_REG="TRUE", keep="true" *) logic [P-1:0] wg1,wg2,rg1,rg2;
 logic full,empty;
 wire [P-1:0] wn=wb+{{(P-1){1'b0}},(wvalid&&!full)},rn=rb+{{(P-1){1'b0}},(rready&&!empty)};
 wire [P-1:0] wgn=(wn>>1)^wn,rgn=(rn>>1)^rn;
 assign wready=!full&&!wrst; assign rvalid=!empty&&!rrst;assign rdata=mem[rb[ADDR-1:0]];
 always_ff @(posedge wclk) if(!wrst&&wvalid&&wready) mem[wb[ADDR-1:0]]<=wdata;
 always_ff @(posedge wclk or posedge wrst) begin
  if(wrst) begin wb<=0;wg<=0;rg1<=0;rg2<=0;full<=0;end
  else begin
   wb<=wn;wg<=wgn;rg1<=rg;rg2<=rg1;
   full<=wgn=={~rg2[P-1:P-2],rg2[P-3:0]};
  end
 end
 always_ff @(posedge rclk or posedge rrst) begin
  if(rrst) begin rb<=0;rg<=0;wg1<=0;wg2<=0;empty<=1;end
  else begin
   rb<=rn;rg<=rgn;wg1<=wg;wg2<=wg1;empty<=rgn==wg2;
  end
 end
`ifndef SYNTHESIS
 initial begin assert(ADDR>=2); assert(WIDTH>0); end
 logic [P-1:0] previous_wg,previous_rg;
 always_ff @(posedge wclk or posedge wrst) begin
   if(wrst) previous_wg<='0;
   else begin assert($onehot0(wg^previous_wg)); previous_wg<=wg; end
 end
 always_ff @(posedge rclk or posedge rrst) begin
   if(rrst) previous_rg<='0;
   else begin assert($onehot0(rg^previous_rg)); previous_rg<=rg; end
 end
`endif
endmodule
