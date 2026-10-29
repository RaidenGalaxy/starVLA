#!/usr/bin/env python3
"""Generate a structurally pipelined FMA from pinned CVFPU combinational blocks.

Unlike NumPipeRegs, these cuts cross the multiplier compression tree. The
upstream operation selection, special cases, normalization and rounding are
retained verbatim (apart from registered signal substitutions).
"""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]

def upstream_notice():
    raw=(ROOT/'vendor/fpnew_fma.sv').read_text()
    return raw[:raw.index('`include')]

def csa_layers(names, width, stop, declarations, statements, prefix):
    level=0
    while len(names)>stop:
        out=[]
        for i in range(0,len(names)-2,3):
            a,b,c=names[i:i+3]; s=f'{prefix}_{level}_{i}_s'; t=f'{prefix}_{level}_{i}_c'
            declarations.append(f'logic [{width}-1:0] {s}, {t};')
            statements += [f'assign {s} = {a} ^ {b} ^ {c};', f'assign {t} = (({a}&{b}) | ({a}&{c}) | ({b}&{c})) << 1;']
            out += [s,t]
        out += names[len(names)//3*3:]; names=out; level+=1
    return names

def product_module(w,fast=False):
    d=[]; s=[]; names=[]
    for i in range(w):
        n=f'p{i}'; d.append(f'logic [{2*w-1}:0] {n};')
        s.append(f'assign {n} = ({w}\'d0 + {{ {w}{{b[{i}]}} }} & a) << {i};')
        # Explicitly extend before shifting; Verilog context widths are not relied on.
        s[-1]=f'assign {n} = {{{{{w}{{1\'b0}}}}, (a & {{{w}{{b[{i}]}}}})}} << {i};'
        names.append(n)
    names=csa_layers(names,str(2*w),max(2,w//4),d,s,'a')
    regs=[]
    for i,n in enumerate(names):
        q=f'q{i}';d.append(f'logic [{2*w-1}:0] {q};');regs.append(f'{q} <= {n};')
    names=[f'q{i}' for i in range(len(names))]
    names=csa_layers(names,str(2*w),2,d,s,'b')
    d.append(f'logic [{2*w-1}:0] r0,r1;')
    prefix_name='Fast' if fast else ''
    extra=f"logic [{2*w}:0] final_add;\nSimdPrefixAdd #(.W({2*w})) cpa(.a(r0),.b(r1),.cin(1'b0),.result(final_add));" if fast else ''
    final_sum=f'final_add[{2*w-1}:0]' if fast else 'r0 + r1'
    return f'''module SimdProduct{prefix_name}{w}(input logic clock, ce,
 input logic [{w-1}:0] a,b, output logic [{2*w-1}:0] product);
{chr(10).join(d)}
{chr(10).join(s)}
{extra}
always_ff @(posedge clock) if(ce) begin
 {chr(10).join(regs)}
 r0 <= {names[0]}; r1 <= {names[1]};
 product <= {final_sum};
end
endmodule
'''

def hf_module(width, fmt):
    raw=(ROOT/'vendor/fpnew_fma.sv').read_text()
    constants=raw[raw.index('  localparam int unsigned EXP_BITS'):raw.index('  // Pipelines')]
    constants += '''typedef struct packed {
 logic sign; logic [EXP_BITS-1:0] exponent; logic [MAN_BITS-1:0] mantissa;
} fp_t;
'''
    select=raw[raw.index('  fpnew_pkg::fp_info_t [2:0] info_q;'):raw.index('  logic any_operand_inf;')]
    select=select.replace('inp_pipe_operands_q[NUM_INP_REGS]','operands')
    select=select.replace('inp_pipe_is_boxed_q[NUM_INP_REGS]',"3'b111")
    select=select.replace('inp_pipe_op_mod_q[NUM_INP_REGS]','op_mod')
    select=select.replace('inp_pipe_op_q[NUM_INP_REGS]','operation')
    select=select.replace('inp_pipe_rnd_mode_q[NUM_INP_REGS]','fpnew_pkg::RNE')
    exp=raw[raw.index('  logic any_operand_inf;'):raw.index('  // Product data path')]
    for n in ['operand_a','operand_b','operand_c','info_a','info_b','info_c']:
        exp=re.sub(r'\b'+n+r'\b','s1_'+n,exp)
    norm=raw[raw.index('  logic        [LOWER_SUM_WIDTH-1:0]  sum_lower;'):raw.index('  // Rounding and classification')]
    rounding=raw[raw.index('  logic                         pre_round_sign;'):raw.index('  // Output Pipeline')]
    replace={'final_exponent':'n6.exp','final_mantissa':'n6.mant','sticky_after_norm':'n6.sticky',
             'sum_sticky_bits':'n6.sbits','final_sign_q':'n6.sign','effective_subtraction_q':'n6.sub',
             'result_is_special_q':'n6.special','special_result_q':'n6.sresult','special_status_q':'n6.status',
             'rnd_mode_q':'fpnew_pkg::RNE'}
    for a,b in replace.items(): rounding=re.sub(r'\b'+a+r'\b',b,rounding)
    w=width; p=53 if w==64 else 24
    notice=raw[:raw.index('`include')]
    return notice+f'''// Derived from vendor/fpnew_fma.sv; arithmetic register boundaries modified.
module SimdFpHF{w}(input logic clock, ce, input logic [{w-1}:0] a,b,c,
 input logic [2:0] op, output logic [{w-1}:0] result,
 output logic [4:0] flags);
localparam fpnew_pkg::fp_format_e FpFormat=fpnew_pkg::{fmt};
localparam int WIDTH={w};
{constants}
logic [2:0][WIDTH-1:0] operands;
fpnew_pkg::operation_e operation;
logic op_mod;
always_comb begin
 operands={{c,b,a}}; operation=fpnew_pkg::FMADD; op_mod=1'b0;
 case(op)
 0: operation=fpnew_pkg::MUL;
 2,3: begin operation=fpnew_pkg::ADD; operands={{b,a,{{1'b0,{{(WIDTH-1){{1'b0}}}}}}}}; op_mod=op==3; end
 default: ;
 endcase
end
{select}
fp_t s1_operand_a,s1_operand_b,s1_operand_c;
fpnew_pkg::fp_info_t s1_info_a,s1_info_b,s1_info_c;
always_ff @(posedge clock) if(ce) begin
 s1_operand_a<=operand_a; s1_operand_b<=operand_b; s1_operand_c<=operand_c;
 s1_info_a<=info_a; s1_info_b<=info_b; s1_info_c<=info_c;
end
{exp}
typedef struct packed {{
 logic [PRECISION_BITS-1:0] mc;
 logic signed [EXP_WIDTH-1:0] ep,ed,et;
 logic [SHIFT_AMOUNT_WIDTH-1:0] sh;
 logic sub,sign,special; fp_t sresult; fpnew_pkg::status_t status;
}} meta_t;
meta_t m2,m3,m4;
always_ff @(posedge clock) if(ce) begin
 m2.mc<={{s1_info_c.is_normal,s1_operand_c.mantissa}};
 m2.ep<=exponent_product; m2.ed<=exponent_difference; m2.et<=tentative_exponent;
 m2.sh<=addend_shamt; m2.sub<=effective_subtraction; m2.sign<=tentative_sign;
 m2.special<=result_is_special; m2.sresult<=special_result; m2.status<=special_status;
 m3<=m2; m4<=m3;
end
logic [2*PRECISION_BITS-1:0] product;
SimdProduct{p} mult(.clock(clock),.ce(ce),
 .a({{s1_info_a.is_normal,s1_operand_a.mantissa}}),
 .b({{s1_info_b.is_normal,s1_operand_b.mantissa}}),.product(product));
logic [3*PRECISION_BITS+3:0] product_shifted,addend_after_shift,addend_shifted;
logic [PRECISION_BITS-1:0] addend_sticky_bits;
logic sticky_before_add,inject_carry_in;
assign product_shifted=product<<2;
assign {{addend_after_shift,addend_sticky_bits}} =
 ({{m4.mc,{{(3*PRECISION_BITS+4){{1'b0}}}}}} >> m4.sh);
assign sticky_before_add=|addend_sticky_bits;
assign addend_shifted=m4.sub ? ~addend_after_shift : addend_after_shift;
assign inject_carry_in=m4.sub & ~sticky_before_add;
logic [3*PRECISION_BITS+4:0] sum_pos,sum_neg;
logic [3*PRECISION_BITS+3:0] sum;
logic sum_carry,final_sign;
assign sum_pos=product_shifted+addend_shifted+inject_carry_in;
assign sum_neg=addend_after_shift-product_shifted;
assign sum_carry=sum_pos[3*PRECISION_BITS+4];
assign sum=(m4.sub && !sum_carry) ? sum_neg : sum_pos;
assign final_sign=(m4.sub && (sum_carry==m4.sign)) ? 1'b1 : (m4.sub ? 1'b0 : m4.sign);
meta_t m5;
logic effective_subtraction_q,final_sign_q,sticky_before_add_q;
logic signed [EXP_WIDTH-1:0] exponent_product_q,exponent_difference_q,tentative_exponent_q;
logic [SHIFT_AMOUNT_WIDTH-1:0] addend_shamt_q;
logic [3*PRECISION_BITS+3:0] sum_q;
always_ff @(posedge clock) if(ce) begin
 m5<=m4; sum_q<=sum; final_sign_q<=final_sign; sticky_before_add_q<=sticky_before_add;
end
assign effective_subtraction_q=m5.sub;
assign exponent_product_q=m5.ep; assign exponent_difference_q=m5.ed;
assign tentative_exponent_q=m5.et; assign addend_shamt_q=m5.sh;
{norm}
typedef struct packed {{
 logic signed [EXP_WIDTH-1:0] exp;
 logic [PRECISION_BITS:0] mant; logic [2*PRECISION_BITS+2:0] sbits;
 logic sign,sub,sticky,special; fp_t sresult; fpnew_pkg::status_t status;
}} norm_t;
norm_t n6;
always_ff @(posedge clock) if(ce) begin
 n6.exp<=final_exponent; n6.mant<=final_mantissa; n6.sbits<=sum_sticky_bits;
 n6.sign<=final_sign_q; n6.sub<=m5.sub; n6.sticky<=sticky_after_norm;
 n6.special<=m5.special; n6.sresult<=m5.sresult; n6.status<=m5.status;
end
{rounding}
logic [WIDTH-1:0] result7; logic [4:0] flags7;
always_ff @(posedge clock) if(ce) begin
 result7<=result_d; flags7<=status_d; result<=result7; flags<=flags7;
end
endmodule
'''

def main():
    output=upstream_notice()+'// Modified: explicit SIMD arithmetic pipeline registers and carry trees.\n'
    output+='// Generated by scripts/generate_fp.py; do not edit this file.\n'
    output+='\n'.join(product_module(w) for w in (24,53,64))
    output+='\n'.join(hf_module(w,f) for w,f in ((32,'FP32'),(64,'FP64')))
    (ROOT/'rtl/SimdFpHF.sv').write_text(output)

if __name__=='__main__': raise SystemExit('HF generation is retired; use scripts/generate_fp2.py')
