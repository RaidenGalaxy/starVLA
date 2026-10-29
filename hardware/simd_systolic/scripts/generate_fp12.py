#!/usr/bin/env python3
"""Twelve real FMA cuts: early C alignment, split sum/sign and coarse/fine norm."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
from generate_fp import hf_module,product_module,ROOT,upstream_notice

def make(width,fmt):
    s=hf_module(width,fmt).replace(f'module SimdFpHF{width}',f'module SimdFpHF12_{width}')
    # CTS identifies S1 -> product exponent -> exponent difference -> maximum
    # exponent as a critical arithmetic chain. Compute the biased exponents
    # in S1, then compute the difference and alignment shift independently in
    # S2. This moves real logic across an existing cut without adding latency.
    start=s.index('  // Zero-extend exponents into signed container')
    end=s.index('typedef struct packed {\n logic [PRECISION_BITS-1:0] mc;',start)
    s=s[:start]+'''logic [EXP_WIDTH-1:0] exponent_a0,exponent_b0,exponent_c0;
localparam logic [EXP_WIDTH-1:0] NegativeBias=EXP_WIDTH'(-$signed(BIAS));
logic [EXP_WIDTH-1:0] exponent_sum0,exponent_carry0;
logic [EXP_WIDTH:0] exponent_product_full0,alignment_base_full0;
logic signed [EXP_WIDTH-1:0] alignment_base1;
// A subnormal has encoded exponent zero; OR inserts its effective exponent
// one without putting an incrementer before the product exponent adder.
assign exponent_a0=EXP_WIDTH'(operand_a.exponent)|EXP_WIDTH'(info_a.is_subnormal);
assign exponent_b0=EXP_WIDTH'(operand_b.exponent)|EXP_WIDTH'(info_b.is_subnormal);
assign exponent_c0=info_c.is_normal ? EXP_WIDTH'(operand_c.exponent) :
 (operand_c.exponent=='0 ? EXP_WIDTH'(1) : EXP_WIDTH'(1<<EXP_BITS));
assign exponent_sum0=exponent_a0^exponent_b0^NegativeBias;
assign exponent_carry0=((exponent_a0&exponent_b0)|(exponent_a0&NegativeBias)|(exponent_b0&NegativeBias))<<1;
SimdPrefixAdd #(.W(EXP_WIDTH)) product_exponent0(
 .a(exponent_sum0),.b(exponent_carry0),.cin(1'b0),.result(exponent_product_full0));
SimdPrefixAdd #(.W(EXP_WIDTH)) alignment_base0(
 .a(EXP_WIDTH'(PRECISION_BITS+3)),.b(~exponent_c0),.cin(1'b1),.result(alignment_base_full0));
always_ff @(posedge clock) if(ce) begin
 exponent_product <= info_a.is_zero||info_b.is_zero ? EXP_WIDTH'(2-$signed(BIAS)) : exponent_product_full0[EXP_WIDTH-1:0];
 exponent_addend <= exponent_c0;
 alignment_base1 <= alignment_base_full0[EXP_WIDTH-1:0];
end
logic [EXP_WIDTH:0] exponent_difference_full,alignment_full;
logic signed [EXP_WIDTH-1:0] alignment_shift;
SimdPrefixAdd #(.W(EXP_WIDTH)) exponent_difference_add(
 .a(exponent_addend),.b(~exponent_product),.cin(1'b1),.result(exponent_difference_full));
assign exponent_difference=$signed(exponent_difference_full[EXP_WIDTH-1:0]);
// Equality can select either equal exponent; inspecting the sign avoids a
// zero-detection reduction after the subtractor.
assign tentative_exponent=exponent_difference[EXP_WIDTH-1] ? exponent_product : exponent_addend;
SimdPrefixAdd #(.W(EXP_WIDTH)) alignment_add(
 .a(exponent_product),.b(alignment_base1),.cin(1'b0),.result(alignment_full));
assign alignment_shift=$signed(alignment_full[EXP_WIDTH-1:0]);
logic [SHIFT_AMOUNT_WIDTH-1:0] addend_shamt;
always_comb begin
 if(alignment_shift<0) addend_shamt='0;
 else if(alignment_shift>=EXP_WIDTH'(3*PRECISION_BITS+4)) addend_shamt=SHIFT_AMOUNT_WIDTH'(3*PRECISION_BITS+4);
 else addend_shamt=alignment_shift[SHIFT_AMOUNT_WIDTH-1:0];
end

'''+s[end:]
    s=s.replace('logic sub,sign,special; fp_t sresult;',
        'logic sub,sign,special; logic [3*PRECISION_BITS+3:0] aligned; logic csticky; fp_t sresult;')
    s=s.replace('meta_t m2,m3,m4;', '''meta_t m2,m3,m4;
logic [4*PRECISION_BITS+3:0] shifted_c2,coarse_c3,shifted_c3;
// Keep the complete addend through both shifts. Sticky reduction occurs
// only after the fine shift, so no information or rounding point is lost.
logic [63:0] align_coarse_controls2,align_fine_controls3;
SimdShiftControlReplicas align_coarse_registers2(.clock(clock),.ce(ce),
 .data(4'(addend_shamt>>4)),.value(align_coarse_controls2));
SimdShiftControlReplicas align_fine_registers3(.clock(clock),.ce(ce),
 .data(m2.sh[3:0]),.value(align_fine_controls3));
for(genvar chunk=0;chunk<(4*PRECISION_BITS+4+15)/16;chunk++)begin: local_alignment
 localparam int Low=16*chunk;
 localparam int Size=(4*PRECISION_BITS+4-Low)>16?16:(4*PRECISION_BITS+4-Low);
 wire [4*PRECISION_BITS+3:0] coarse_word={m2.mc,{(3*PRECISION_BITS+4){1'b0}}}>>{align_coarse_controls2[4*chunk+:4],4'b0};
 wire [4*PRECISION_BITS+3:0] fine_word=coarse_c3>>align_fine_controls3[4*chunk+:4];
 assign shifted_c2[Low+:Size]=coarse_word[Low+:Size];
 assign shifted_c3[Low+:Size]=fine_word[Low+:Size];
end''')
    s=s.replace(' m3<=m2; m4<=m3;', ''' m2.aligned<='0; m2.csticky<=0;
 m3<=m2;coarse_c3<=shifted_c2;m4<=m3;
 m4.aligned<=shifted_c3[4*PRECISION_BITS+3:PRECISION_BITS];
 m4.csticky<=|shifted_c3[PRECISION_BITS-1:0];''')
    start=s.index('assign {addend_after_shift,addend_sticky_bits}')
    end=s.index('assign addend_shifted=',start)
    s=s[:start]+'''assign addend_after_shift=m4.aligned;
assign addend_sticky_bits='0;
assign sticky_before_add=m4.csticky;
'''+s[end:]
    # S4 was a metadata relay beside the final product CPA. Prepare the
    # addend complement/carry here, so S5's wide prefix add does not start
    # with a high-fanout subtraction selector.
    s=s.replace('assign addend_shifted=m4.sub ? ~addend_after_shift : addend_after_shift;\nassign inject_carry_in=m4.sub & ~sticky_before_add;', '''logic [3*PRECISION_BITS+3:0] prepared_addend4;
logic prepared_carry4;
always_ff @(posedge clock) if(ce) begin
 prepared_addend4<=m3.sub ? ~shifted_c3[4*PRECISION_BITS+3:PRECISION_BITS] : shifted_c3[4*PRECISION_BITS+3:PRECISION_BITS];
 prepared_carry4<=m3.sub & ~(|shifted_c3[PRECISION_BITS-1:0]);
end
assign addend_shifted=prepared_addend4;
assign inject_carry_in=prepared_carry4;''')
    start=s.index('meta_t m5;');end=s.index('  logic        [LOWER_SUM_WIDTH-1:0]  sum_lower;',start)
    s=s[:start]+'''meta_t m5,m6;
logic [3*PRECISION_BITS+4:0] positive5,negative5;
logic sticky5;
logic effective_subtraction_q,final_sign_q,sticky_before_add_q;
logic signed [EXP_WIDTH-1:0] exponent_product_q,exponent_difference_q,tentative_exponent_q;
logic [SHIFT_AMOUNT_WIDTH-1:0] addend_shamt_q;
logic [3*PRECISION_BITS+3:0] sum_q;
wire carry5=positive5[3*PRECISION_BITS+4];
always_ff @(posedge clock) if(ce) begin
 m5<=m4;positive5<=sum_pos;negative5<=sum_neg;sticky5<=sticky_before_add;
 m6<=m5;sum_q<=m5.sub&&!carry5 ? negative5 : positive5;
 final_sign_q <= (m5.sub&&(carry5==m5.sign)) ? 1'b1 : (m5.sub ? 1'b0 : m5.sign);
 sticky_before_add_q<=sticky5;
end
assign effective_subtraction_q=m6.sub;
assign exponent_product_q=m6.ep;assign exponent_difference_q=m6.ed;
assign tentative_exponent_q=m6.et;assign addend_shamt_q=m6.sh;
'''+s[end:]
    s=s.replace('  assign sum_shifted       = sum_q << norm_shamt;', '''
logic [3*PRECISION_BITS+3:0] norm_sum7;
logic [SHIFT_AMOUNT_WIDTH-1:0] norm_shift7;
logic signed [EXP_WIDTH-1:0] norm_exp7,norm_exp8;
logic [3*PRECISION_BITS+4:0] coarse8;
logic [3:0] low_shift8;
logic norm_sign7,norm_sign8,norm_sticky7,norm_sticky8;
meta_t m7,m8;
always_ff @(posedge clock) if(ce) begin
 norm_sum7<=sum_q;norm_shift7<=norm_shamt;norm_exp7<=normalized_exponent;
 norm_sign7<=final_sign_q;norm_sticky7<=sticky_before_add_q;m7<=m6;
 coarse8<={1'b0,norm_sum7} << {norm_shift7[SHIFT_AMOUNT_WIDTH-1:4],4'b0};
 low_shift8<=norm_shift7[3:0];norm_exp8<=norm_exp7;
 norm_sign8<=norm_sign7;norm_sticky8<=norm_sticky7;m8<=m7;
end
assign sum_shifted=coarse8<<low_shift8;''')
    # The S8 shift amount drives a full-width fine barrel shifter. CTS shows
    # that its shared control FF load dominates this cut; use kept local
    # register copies without moving the S8/S9 boundary or rounding point.
    s=s.replace('low_shift8<=norm_shift7[3:0];norm_exp8<=norm_exp7;',
                'norm_exp8<=norm_exp7;')
    s=s.replace('assign sum_shifted=coarse8<<low_shift8;', '''logic [63:0] fine_shift_controls8;
SimdShiftControlReplicas fine_shift_registers8(.clock(clock),.ce(ce),
 .data(norm_shift7[3:0]),.value(fine_shift_controls8));
assign low_shift8=fine_shift_controls8[3:0];
for(genvar word=0;word<(3*PRECISION_BITS+5+15)/16;word++) begin: fine_normalization
 localparam int Lo=16*word;
 localparam int Chunk=(3*PRECISION_BITS+5-Lo)<16 ? (3*PRECISION_BITS+5-Lo) : 16;
 wire [3*PRECISION_BITS+4:0] shifted=coarse8<<fine_shift_controls8[4*word+:4];
 assign sum_shifted[Lo+:Chunk]=shifted[Lo+:Chunk];
end''')
    # Only the small normalization block consumes the delayed exponent.
    start=s.index('  always_comb begin : small_norm');end=s.index('norm_t n6;',start)
    part=s[start:end].replace('normalized_exponent','norm_exp8').replace('sticky_before_add_q','norm_sticky8')
    s=s[:start]+part+s[end:]
    s=s.replace('n6.sign<=final_sign_q; n6.sub<=m5.sub;', 'n6.sign<=norm_sign8; n6.sub<=m8.sub;')
    s=s.replace('n6.special<=m5.special; n6.sresult<=m5.sresult; n6.status<=m5.status;',
        'n6.special<=m8.special; n6.sresult<=m8.sresult; n6.status<=m8.status;')
    s=s.replace('logic [WIDTH-1:0] result7; logic [4:0] flags7;',
        'logic [WIDTH-1:0] result7,result11; logic [4:0] flags7,flags11;')
    s=s.replace('result<=result7; flags<=flags7;',
        'result11<=result7;flags11<=flags7;result<=result11;flags<=flags11;')
    s=s.replace('assign sum_pos=product_shifted+addend_shifted+inject_carry_in;',
        'SimdPrefixAdd #(.W(3*PRECISION_BITS+4)) positive_add(.a(product_shifted),.b(addend_shifted),.cin(inject_carry_in),.result(sum_pos));')
    s=s.replace('assign sum_neg=addend_after_shift-product_shifted;',
        'logic [3*PRECISION_BITS+4:0] negative_full;\n'
        'SimdPrefixAdd #(.W(3*PRECISION_BITS+4)) negative_add(.a(addend_after_shift),.b(~product_shifted),.cin(1\'b1),.result(negative_full));\n'
        'assign sum_neg={1\'b0,negative_full[3*PRECISION_BITS+3:0]};')
    s=s.replace('SimdProduct'+str(53 if width==64 else 24)+' mult','SimdProductFast'+str(53 if width==64 else 24)+' mult')
    # S6 computes LZC for both registered sum candidates in parallel. S7 then
    # corrects the exponent without concatenating a wide LZC and an adder.
    start=s.index('  // Leading zero counter for cancellations')
    end=s.index('  assign leading_zero_count_sgn',start)
    s=s[:start]+'''logic [LZC_RESULT_WIDTH-1:0] positive_lz5,negative_lz5;
logic positive_empty5,negative_empty5;
logic signed [EXP_WIDTH-1:0] exponent_plus1_6,norm_exp_candidate;
logic [EXP_WIDTH:0] norm_exp_full;
lzc #(.WIDTH(LOWER_SUM_WIDTH),.MODE(1)) positive_lzc5(
 .in_i(positive5[LOWER_SUM_WIDTH-1:0]),.cnt_o(positive_lz5),.empty_o(positive_empty5));
lzc #(.WIDTH(LOWER_SUM_WIDTH),.MODE(1)) negative_lzc5(
 .in_i(negative5[LOWER_SUM_WIDTH-1:0]),.cnt_o(negative_lz5),.empty_o(negative_empty5));
always_ff @(posedge clock) if(ce) begin
 leading_zero_count <= m5.sub&&!carry5 ? negative_lz5 : positive_lz5;
 lzc_zeroes <= m5.sub&&!carry5 ? negative_empty5 : positive_empty5;
 exponent_plus1_6 <= m5.ep + EXP_WIDTH'(1);
end
SimdPrefixAdd #(.W(EXP_WIDTH)) normalization_exponent(
 .a(exponent_plus1_6),.b(~{{(EXP_WIDTH-LZC_RESULT_WIDTH){1'b0}},leading_zero_count}),
 .cin(1'b1),.result(norm_exp_full));
assign norm_exp_candidate=$signed(norm_exp_full[EXP_WIDTH-1:0]);
'''+s[end:]
    s=s.replace('exponent_product_q - leading_zero_count_sgn + 1','norm_exp_candidate')
    # RNE carry uses an explicit prefix network. Move the S11 padding register
    # into the rounding/status cone, so S10 rounds and S11 classifies/selects.
    start=s.index('  fpnew_rounding #(')
    end=s.index('  // Classification after rounding',start)
    s=s[:start]+"""logic [EXP_BITS+MAN_BITS:0] round_full;
wire round_up=round_sticky_bits[1]&&(round_sticky_bits[0]||pre_round_abs[0]);
SimdPrefixAdd #(.W(EXP_BITS+MAN_BITS)) round_add(
 .a(pre_round_abs),.b('0),.cin(round_up),.result(round_full));
assign rounded_abs=round_full[EXP_BITS+MAN_BITS-1:0];
assign result_zero=(pre_round_abs=='0)&&(round_sticky_bits=='0);
assign rounded_sign=(result_zero&&n6.sub)?1'b0:pre_round_sign;
logic [EXP_BITS+MAN_BITS-1:0] rounded_abs10,pre_round10;
logic rounded_sign10,of_before10;
logic [1:0] round_sticky10;
norm_t norm10;
always_ff @(posedge clock) if(ce) begin
 rounded_abs10<=rounded_abs;rounded_sign10<=rounded_sign;
 pre_round10<=pre_round_abs;of_before10<=of_before_round;
 round_sticky10<=round_sticky_bits;norm10<=n6;
end
"""+s[end:]
    start=s.index('  // Classification after rounding')
    tail=s[start:]
    import re
    for old,new in [('rounded_abs','rounded_abs10'),('rounded_sign','rounded_sign10'),
                    ('pre_round_abs','pre_round10'),('of_before_round','of_before10'),
                    ('round_sticky_bits','round_sticky10'),('n6','norm10')]:
        tail=re.sub(r'\b'+old+r'\b',new,tail)
    s=s[:start]+tail
    s=s.replace('result7<=result_d; flags7<=status_d; result11<=result7;flags11<=flags7;',
                'result11<=result_d;flags11<=status_d;')
    # n6 is the S9 normalization register; result7 is S10 (upstream names retained).
    return s

if __name__=='__main__':
    (ROOT/'rtl/SimdFpHF12.sv').write_text(upstream_notice()+
        '// Modified: twelve arithmetic cuts, prefix carry and local shift controls.\n'+
        '// Generated by scripts/generate_fp12.py\n'+
        '\n'.join(product_module(w,fast=True)for w in (24,53))+
        '\n'.join(make(w,f)for w,f in [(32,'FP32'),(64,'FP64')]))
