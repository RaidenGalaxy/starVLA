// SPDX-License-Identifier: Apache-2.0
// Memory-domain output conversion. Normally converts one FP32 value to a
// runtime destination. wide_source first folds an FP64 accumulator to FP32.
// Integer outputs are RNE(value) + signed zero_point, then saturate. FP4 is
// OCP E2M1 with finite saturation, signed zero, and code-parity RNE ties.
module SimdOutputConvert(input logic [63:0] value,input logic wide_source,
 input logic [3:0] format,input logic [31:0] zero_point,
 output logic [63:0] result,output logic [4:0] flags,output logic saturated);
 logic [3:0] fmt;
 assign fmt=wide_source?4'd9:format;
 fpnew_pkg::fp_format_e destination;
 always_comb begin
  destination=fpnew_pkg::FP32;
  case(fmt)
   6:destination=fpnew_pkg::FP8;
   7:destination=fpnew_pkg::FP16;
   8:destination=fpnew_pkg::FP16ALT;
   10:destination=fpnew_pkg::FP64;
   default:begin end
  endcase
 end
 logic [63:0] cast_result;
 logic [4:0] cast_flags;
 fpnew_cast_multi #(.FpFmtConfig(5'b11111),.IntFmtConfig(4'b0010),.NumPipeRegs(0)) cast (
  .clk_i(1'b0),.rst_ni(1'b1),.operands_i(value),.is_boxed_i('1),
  .rnd_mode_i(fpnew_pkg::RNE),.op_i(fpnew_pkg::F2F),.op_mod_i(1'b0),
  .src_fmt_i(wide_source?fpnew_pkg::FP64:fpnew_pkg::FP32),.dst_fmt_i(destination),
  .int_fmt_i(fpnew_pkg::INT32),.tag_i(1'b0),.mask_i(1'b1),.aux_i(1'b0),
  .in_valid_i(1'b1),.in_ready_o(),.flush_i(1'b0),.result_o(cast_result),
  .status_o(cast_flags),.extension_bit_o(),.tag_o(),.mask_o(),.aux_o(),
  .out_valid_o(),.out_ready_i(1'b1),.busy_o(),.reg_ena_i('0),.early_out_valid_o());

 wire sign_bit=value[31];
 wire [30:0] absolute=value[30:0];
 wire [7:0] exponent=value[30:23];
 wire [22:0] fraction=value[22:0];
 wire is_nan=(&exponent)&&(|fraction);
 wire is_inf=(&exponent)&&!(|fraction);
 wire [23:0] significand={exponent!=0,fraction};
 integer power,shift;
 logic [65:0] magnitude;
 logic [23:0] kept,lost,low_mask;
 logic discarded,guard_bit,sticky,increment;
 always_comb begin
  power=exponent==0?-126:integer'(exponent)-127;
  shift=0;magnitude=0;kept=0;lost=0;low_mask=0;
  discarded=0;guard_bit=0;sticky=0;increment=0;
  if((&exponent)||power>=65)magnitude=66'd1<<65;
  else if(power>=23)magnitude={42'd0,significand}<<(power-23);
  else if(power>=0)begin
   shift=23-power;kept=significand>>shift;
   low_mask=(24'd1<<shift)-1'b1;lost=significand&low_mask;
   guard_bit=significand[shift-1];
   sticky=(significand&((24'd1<<(shift-1))-1'b1))!=0;
   increment=guard_bit&&(sticky||kept[0]);
   magnitude={42'd0,kept}+increment;discarded=lost!=0;
  end else begin
   // Exactly +/-0.5 ties to zero; larger magnitudes below one round to one.
   magnitude=power==-1&&significand>24'h800000;
   discarded=absolute!=0;
  end
 end
 logic signed [66:0] integral,adjusted,minimum,maximum,clipped;
 logic [63:0] integer_mask,integer_result;
 logic integer_clipped;
 logic [4:0] integer_flags;
 always_comb begin
  integral=sign_bit?-$signed({1'b0,magnitude}):$signed({1'b0,magnitude});
  if(is_nan)integral=0;
  adjusted=integral+$signed(zero_point);
  minimum=-(67'sd1<<63);maximum=(67'sd1<<63)-1'b1;integer_mask='1;
  case(fmt)
   0:begin minimum=-8;maximum=7;integer_mask=64'hf;end
   1:begin minimum=-128;maximum=127;integer_mask=64'hff;end
   2:begin minimum=-32768;maximum=32767;integer_mask=64'hffff;end
   3:begin minimum=-(67'sd1<<31);maximum=(67'sd1<<31)-1'b1;integer_mask=64'hffffffff;end
   default:begin end
  endcase
  integer_clipped=adjusted<minimum||adjusted>maximum;
  clipped=adjusted<minimum?minimum:adjusted>maximum?maximum:adjusted;
  integer_result=clipped[63:0]&integer_mask;
  integer_flags=(is_nan||integer_clipped)?5'b10000:discarded?5'b00001:5'b00000;
 end

 function automatic logic [31:0] fp4_value(input logic [2:0] code);
  case(code)
   0:return 32'h00000000;1:return 32'h3f000000;
   2:return 32'h3f800000;3:return 32'h3fc00000;
   4:return 32'h40000000;5:return 32'h40400000;
   6:return 32'h40800000;7:return 32'h40c00000;
  endcase
 endfunction
 logic [2:0] fp4_code;
 always_comb begin
  if(absolute<=32'h3e800000)fp4_code=0;
  else if(absolute<32'h3f400000)fp4_code=1;
  else if(absolute<=32'h3fa00000)fp4_code=2;
  else if(absolute<32'h3fe00000)fp4_code=3;
  else if(absolute<=32'h40200000)fp4_code=4;
  else if(absolute<32'h40600000)fp4_code=5;
  else if(absolute<=32'h40a00000)fp4_code=6;
  else fp4_code=7;
  result=0;flags=0;saturated=0;
  if(fmt<=4)begin result=integer_result;flags=integer_flags;saturated=integer_clipped;end
  else if(fmt==5)begin
   result={60'd0,sign_bit,fp4_code};
   saturated=absolute>32'h40c00000&&!is_nan;
   flags=saturated?5'b00101:absolute!=fp4_value(fp4_code)?5'b00001:5'b00000;
   if(is_nan)begin result=0;flags=5'b10000;saturated=0;end
  end else begin
   flags=cast_flags;
   case(fmt)
    6:result={56'd0,cast_result[7:0]};
    7,8:result={48'd0,cast_result[15:0]};
    9:result={32'd0,cast_result[31:0]};
    10:result=cast_result;
    default:begin result=0;flags=5'b10000;end
   endcase
  end
 end
endmodule
