module ResponseFifoProbe(input logic clock,reset,input logic [3:0] write,read,
 input logic [127:0] data,output logic [127:0] value);
 SimdResponseBank10 q0(.clock(clock),.reset(reset),.write(write[0]),.ready(read[0]),.data(data[31:0]),.value(value[31:0]));
 SimdResponseBank11 q1(.clock(clock),.reset(reset),.write(write[1]),.ready(read[1]),.data(data[63:32]),.value(value[63:32]));
 SimdResponseBank14 q2(.clock(clock),.reset(reset),.write(write[2]),.ready(read[2]),.data(data[95:64]),.value(value[95:64]));
 SimdResponseBank15 q3(.clock(clock),.reset(reset),.write(write[3]),.ready(read[3]),.data(data[127:96]),.value(value[127:96]));
endmodule
