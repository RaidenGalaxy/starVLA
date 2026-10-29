// SPDX-License-Identifier: Apache-2.0
// Optional Nangate45 implementation of positive-edge held/reset controls.
// SDFF_X1 is a scan-capable flop, used here as an ordinary functional flop:
// its internal selection implements the original enable or synchronous reset.
// Other FF types retain the existing dfflibmap implementation.
module \$_DFFE_PP_ (input D,C,E,output Q);
  SDFF_X1 _TECHMAP_REPLACE_ (.CK(C),.D(Q),.SE(E),.SI(D),.Q(Q),.QN());
endmodule
module \$_SDFF_PP0_ (input D,C,R,output Q);
  SDFF_X1 _TECHMAP_REPLACE_ (.CK(C),.D(D),.SE(R),.SI(1'b0),.Q(Q),.QN());
endmodule
module \$_SDFF_PP1_ (input D,C,R,output Q);
  SDFF_X1 _TECHMAP_REPLACE_ (.CK(C),.D(D),.SE(R),.SI(1'b1),.Q(Q),.QN());
endmodule
