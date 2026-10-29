// SPDX-License-Identifier: Apache-2.0
// Memory-domain conversion used after FP32 weight dequantization. The pinned
// CVFPU cast performs RNE with gradual underflow and IEEE exception flags.
module SimdNarrow32(input logic [31:0] value,input logic bf16,
                    output logic [15:0] result,output logic [4:0] flags);
 logic [31:0] boxed_result;
 fpnew_cast_multi #(.FpFmtConfig(5'b10101),.IntFmtConfig(4'b0010),.NumPipeRegs(0)) cast (
   .clk_i(1'b0),.rst_ni(1'b1),.operands_i(value),.is_boxed_i('1),
   .rnd_mode_i(fpnew_pkg::RNE),.op_i(fpnew_pkg::F2F),.op_mod_i(1'b0),
   .src_fmt_i(fpnew_pkg::FP32),.dst_fmt_i(bf16?fpnew_pkg::FP16ALT:fpnew_pkg::FP16),
   .int_fmt_i(fpnew_pkg::INT32),.tag_i(1'b0),.mask_i(1'b1),.aux_i(1'b0),
   .in_valid_i(1'b1),.in_ready_o(),.flush_i(1'b0),.result_o(boxed_result),
   .status_o(flags),.extension_bit_o(),.tag_o(),.mask_o(),.aux_o(),
   .out_valid_o(),.out_ready_i(1'b1),.busy_o(),.reg_ena_i('0),.early_out_valid_o());
 assign result=boxed_result[15:0];
endmodule
