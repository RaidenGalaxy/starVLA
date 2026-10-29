// Functional contract of ASAP7 srambank_256x4x64_6t122, upstream revision
// 522eeccbccefcd66e61893fa1059df24d95e9f86 (BSD-3-Clause).
// This model supplies no physical certification. Synthesis retains a hard cell.
`ifdef SYNTHESIS
(* blackbox *)
`endif
module srambank_256x4x64_6t122 (
  input clk, input [9:0] ADDRESS, input [63:0] wd,
  input banksel, input read, input write, output reg [63:0] dataout
);
`ifndef SYNTHESIS
  reg [63:0] mem [0:1023];
  always @(posedge clk) begin
    assert (!(banksel && read && write));
    if (banksel && write) mem[ADDRESS] <= wd;
    else if (banksel && read) dataout <= mem[ADDRESS];
  end
`endif
endmodule
