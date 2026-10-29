// SPDX-License-Identifier: Apache-2.0
// F2: exact product/aligned sum in stage 1, normalization/RNE in stage 2.
module SimdFpF232(input logic clock,ce,input logic [31:0] a,b,c,
 input logic [2:0] op,output logic [31:0] result,output logic [4:0] flags);
 logic [31:0] r;logic [4:0] status;
 logic [2:0][31:0] operands; fpnew_pkg::operation_e operation;logic mod;
 always_ff @(posedge clock) if(ce) begin result<=r;flags<=status;end
 always_comb begin
 operands={c,b,a};operation=fpnew_pkg::FMADD;mod=0;
 case(op)
 0:operation=fpnew_pkg::MUL;
 2,3:begin operands={b,a,32'd0};operation=fpnew_pkg::ADD;mod=op==3;end
 default:;
 endcase
 end
 SimdFpF2Core #(.FpFormat(fpnew_pkg::FP32),.NumPipeRegs(1),.PipeConfig(fpnew_pkg::INSIDE)) core(
 .clk_i(clock),.rst_ni(1'b1),.operands_i(operands),.is_boxed_i(3'b111),
 .rnd_mode_i(fpnew_pkg::RNE),.op_i(operation),.op_mod_i(mod),
 .tag_i(1'b0),.mask_i(1'b1),.aux_i(1'b0),.in_valid_i(ce),.in_ready_o(),
 .flush_i(1'b0),.result_o(r),.status_o(status),.extension_bit_o(),.tag_o(),.mask_o(),.aux_o(),
 .out_valid_o(),.out_ready_i(ce),.busy_o(),.reg_ena_i(ce),.early_out_valid_o());
 endmodule
module SimdFpF264(input logic clock,ce,input logic [63:0] a,b,c,
 input logic [2:0] op,output logic [63:0] result,output logic [4:0] flags);
 logic [63:0] r;logic [4:0] status;
 logic [2:0][63:0] operands; fpnew_pkg::operation_e operation;logic mod;
 always_ff @(posedge clock) if(ce) begin result<=r;flags<=status;end
 always_comb begin
 operands={c,b,a};operation=fpnew_pkg::FMADD;mod=0;
 case(op)
 0:operation=fpnew_pkg::MUL;
 2,3:begin operands={b,a,64'd0};operation=fpnew_pkg::ADD;mod=op==3;end
 default:;
 endcase
 end
 SimdFpF2Core #(.FpFormat(fpnew_pkg::FP64),.NumPipeRegs(1),.PipeConfig(fpnew_pkg::INSIDE)) core(
 .clk_i(clock),.rst_ni(1'b1),.operands_i(operands),.is_boxed_i(3'b111),
 .rnd_mode_i(fpnew_pkg::RNE),.op_i(operation),.op_mod_i(mod),
 .tag_i(1'b0),.mask_i(1'b1),.aux_i(1'b0),.in_valid_i(ce),.in_ready_o(),
 .flush_i(1'b0),.result_o(r),.status_o(status),.extension_bit_o(),.tag_o(),.mask_o(),.aux_o(),
 .out_valid_o(),.out_ready_i(ce),.busy_o(),.reg_ena_i(ce),.early_out_valid_o());
 endmodule
