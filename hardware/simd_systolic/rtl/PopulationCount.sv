// SPDX-License-Identifier: Apache-2.0
// Keep each balanced subtotal visible to technology mapping. Flattening the
// equivalent sixteen-input parity into a serial XOR chain caused a measured
// 1 ns event-counter setup violation in the array.
(* keep_hierarchy=1 *)
module SimdPopulationCount #(parameter integer N=16)(
 input wire [N-1:0] bits,
 output wire [$clog2(N+1)-1:0] count);
 localparam integer W=$clog2(N+1);
 if(N==2)begin: pair
  assign count={bits[0]&bits[1],bits[0]^bits[1]};
 end else begin: split
  wire [W-2:0] left,right;
  // Wire keep attributes alone still allowed ABC to reassociate a nine-XOR
  // chain. Preserve every recursive boundary, not only the output names.
  SimdPopulationCount #(.N(N/2)) lo(bits[N/2-1:0],left);
  SimdPopulationCount #(.N(N/2)) hi(bits[N-1:N/2],right);
  assign count={1'b0,left}+{1'b0,right};
 end
endmodule

// Fixed interfaces let the Slang frontend instantiate native-Yosys modules
// without a parameter override on an imported black-box declaration.
module SimdPopulationCount2(input wire [1:0] bits,output wire [1:0] count);
 SimdPopulationCount #(.N(2)) impl(bits,count);
endmodule
module SimdPopulationCount4(input wire [3:0] bits,output wire [2:0] count);
 SimdPopulationCount #(.N(4)) impl(bits,count);
endmodule
module SimdPopulationCount8(input wire [7:0] bits,output wire [3:0] count);
 SimdPopulationCount #(.N(8)) impl(bits,count);
endmodule
module SimdPopulationCount16(input wire [15:0] bits,output wire [4:0] count);
 SimdPopulationCount #(.N(16)) impl(bits,count);
endmodule
