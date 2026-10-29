// SPDX-License-Identifier: Apache-2.0
// Parallel-prefix carry network. Kept prefix boundaries prevent technology
// mapping from collapsing a wide carry path into a ripple chain.
module SimdPrefixAdd #(parameter int W=64)(
 input logic [W-1:0] a,b,input logic cin,output logic [W:0] result);
 localparam int L=$clog2(W);
 (* keep *) wire [W-1:0] p[0:L],g[0:L];
 assign p[0]=a^b;
 assign g[0]=a&b;
 for(genvar s=0;s<L;s++) begin:level
  for(genvar i=0;i<W;i++) begin:bit_group
   if(i>=(1<<s)) begin
    assign p[s+1][i]=p[s][i]&p[s][i-(1<<s)];
    assign g[s+1][i]=g[s][i]|(p[s][i]&g[s][i-(1<<s)]);
   end else begin
    assign p[s+1][i]=p[s][i];assign g[s+1][i]=g[s][i];
   end
  end
 end
 assign result[0]=p[0][0]^cin;
 for(genvar i=1;i<W;i++)
  assign result[i]=p[0][i]^(g[L][i-1]|(p[L][i-1]&cin));
 assign result[W]=g[L][W-1]|(p[L][W-1]&cin);
endmodule
