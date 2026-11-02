// Explicit mapping for the pinned ASAP7 inverted-D/inverted-Q async flop.
// QN = !D on the rising edge; RESETN presets QN, SETN clears QN.
// Logical Q is !QN. Only one asynchronous control is active in these cells.
module \$_DFF_PP0_ (input C,D,R,output Q);
 wire qn;
 DFFASRHQNx1_ASAP7_75t_R ff(.CLK(C),.D(D),.QN(qn),.RESETN(~R),.SETN(1'b1));
 assign Q=~qn;
endmodule
module \$_DFF_PP1_ (input C,D,R,output Q);
 wire qn;
 DFFASRHQNx1_ASAP7_75t_R ff(.CLK(C),.D(D),.QN(qn),.RESETN(1'b1),.SETN(~R));
 assign Q=~qn;
endmodule
module \$_DFF_PN0_ (input C,D,R,output Q);
 wire qn;
 DFFASRHQNx1_ASAP7_75t_R ff(.CLK(C),.D(D),.QN(qn),.RESETN(R),.SETN(1'b1));
 assign Q=~qn;
endmodule
module \$_DFF_PN1_ (input C,D,R,output Q);
 wire qn;
 DFFASRHQNx1_ASAP7_75t_R ff(.CLK(C),.D(D),.QN(qn),.RESETN(1'b1),.SETN(R));
 assign Q=~qn;
endmodule
