#!/usr/bin/env python3
"""Isolated two-stage FP64 experiment; never edits the active RTL sources.

This is a component exploration, not a substitute for the full-format RTL
regression or extracted timing. Outputs retain the exact experimental inputs.
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'physical'), str(ROOT / 'scripts'), str(ROOT / 'python')]
from run import sha, synthesize
from build import rtl_digest
from asap7_platform import profile
from simd_eval.architecture import ARCH
from simd_eval.reference import Format, Op, fp_operation


COMPRESSOR = '''
// Experiment only: unsigned carry-save tree followed by a parallel prefix sum.
// Combinational throughout; the enclosing FMA retains its two registers.
module SimdUnsignedMultiply #(parameter int W=53)(
 input logic [W-1:0] a,b, output logic [2*W-1:0] result);
 localparam int B=2*W;
 wire [B-1:0] partials[W];
 for(genvar i=0;i<W;i++) begin:partial
   assign partials[i]=({{W{1'b0}},a}&{B{b[i]}})<<i;
 end
 SimdCarrySaveReduce #(.B(B),.N(W)) tree(.terms(partials),.result(result));
endmodule
module SimdCarrySaveReduce #(parameter int B=106,N=53)(
 input wire [B-1:0] terms[N],output wire [B-1:0] result);
 if(N==1) begin
   assign result=terms[0];
 end else if(N==2) begin
   wire [B:0] full_result;
   SimdPrefixAdd #(.W(B)) finish(.a(terms[0]),.b(terms[1]),.cin(1'b0),.result(full_result));
   assign result=full_result[B-1:0];
 end else begin
   localparam int GROUPS=N/3,REST=N%3,NEXT=2*GROUPS+REST;
   (* keep *) wire [B-1:0] reduced[NEXT];
   for(genvar i=0;i<GROUPS;i++) begin:compress
     assign reduced[2*i]=terms[3*i]^terms[3*i+1]^terms[3*i+2];
     assign reduced[2*i+1]=((terms[3*i]&terms[3*i+1])|
       (terms[3*i]&terms[3*i+2])|(terms[3*i+1]&terms[3*i+2]))<<1;
   end
   for(genvar i=0;i<REST;i++) assign reduced[2*GROUPS+i]=terms[3*GROUPS+i];
   SimdCarrySaveReduce #(.B(B),.N(NEXT)) next_level(.terms(reduced),.result(result));
 end
endmodule
'''

ONEHOT_NORMALIZATION = '''
  // Experimental normalization: select the leading bit directly. Data and
  // candidate exponents are prepared in parallel instead of serializing a
  // binary LZ count, exponent calculation and a large normalization shifter.
  localparam int NORM_W=3*PRECISION_BITS+4;
  localparam int NORM_T=1<<$clog2(NORM_W);
  localparam int NORM_CW=EXP_WIDTH+PRECISION_BITS+3;
  wire [NORM_W-1:0] norm_choice;
  wire [NORM_CW-1:0] norm_tree[1:2*NORM_T-1];
  for(genvar j=0;j<NORM_T;j++) begin:normal_candidate
    if(j<NORM_W) begin
      wire leading;
      if(j==NORM_W-1) assign leading=sum_q[j];
      else assign leading=sum_q[j] && !(|sum_q[NORM_W-1:j+1]);
      localparam logic signed [EXP_WIDTH-1:0] OFFSET=j;
      wire signed [EXP_WIDTH-1:0] e=exponent_product_q+OFFSET;
      wire [PRECISION_BITS:0] mant;
      wire sticky_top,sticky_rest;
      if(j>=PRECISION_BITS) assign mant=sum_q[j-:PRECISION_BITS+1];
      else assign mant={sum_q[j:0],{(PRECISION_BITS-j){1'b0}}};
      if(j>PRECISION_BITS) assign sticky_top=sum_q[j-PRECISION_BITS-1];
      else assign sticky_top=1'b0;
      if(j>PRECISION_BITS+1) assign sticky_rest=|sum_q[j-PRECISION_BITS-2:0];
      else assign sticky_rest=1'b0;
      assign norm_choice[j]=leading && (e>0);
      assign norm_tree[NORM_T+j]={e,mant,sticky_top,sticky_rest}&{NORM_CW{norm_choice[j]}};
    end else assign norm_tree[NORM_T+j]='0;
  end
  for(genvar node=1;node<NORM_T;node++)
    assign norm_tree[node]=norm_tree[2*node]|norm_tree[2*node+1];
  // The first stage captures the minimum-exponent alignment independently.
  // Only the sticky OR and its highest bit are consumed after normalization;
  // preserve both, including the RISC-V tininess-after-rounding condition.
  wire [NORM_W-1:0] subnormal_vector=sum_q << exponent_difference_q;
  always_comb begin
    final_mantissa='0;final_exponent='0;sum_sticky_bits='0;
    if(|norm_choice) begin
      {final_exponent,final_mantissa,sum_sticky_bits[2*PRECISION_BITS+2],sum_sticky_bits[0]}=norm_tree[1];
    end else begin
      final_mantissa=subnormal_vector[NORM_W-1:2*PRECISION_BITS+3];
      sum_sticky_bits[2*PRECISION_BITS+2]=subnormal_vector[2*PRECISION_BITS+2];
      sum_sticky_bits[0]=|subnormal_vector[2*PRECISION_BITS+1:0];
    end
  end
'''

FUSED_COMPRESSOR = '''
// Retain the product as two carry-save terms through addend injection.
module SimdUnsignedPairMultiply #(parameter int W=53)(
 input wire [W-1:0] a,b,output wire [2*W-1:0] term0,term1);
 localparam int B=2*W;
 wire [B-1:0] partials[W];
 for(genvar i=0;i<W;i++)
   assign partials[i]=({{W{1'b0}},a}&{B{b[i]}})<<i;
 SimdCarrySavePair #(.B(B),.N(W)) tree(.terms(partials),.term0(term0),.term1(term1));
endmodule
module SimdCarrySavePair #(parameter int B=106,N=53)(
 input wire [B-1:0] terms[N],output wire [B-1:0] term0,term1);
 if(N==1) begin
   assign term0=terms[0];assign term1='0;
 end else if(N==2) begin
   assign term0=terms[0];assign term1=terms[1];
 end else begin
   localparam int GROUPS=N/3,REST=N%3,NEXT=2*GROUPS+REST;
   (* keep *) wire [B-1:0] reduced[NEXT];
   for(genvar i=0;i<GROUPS;i++) begin
     assign reduced[2*i]=terms[3*i]^terms[3*i+1]^terms[3*i+2];
     assign reduced[2*i+1]=((terms[3*i]&terms[3*i+1])|
       (terms[3*i]&terms[3*i+2])|(terms[3*i+1]&terms[3*i+2]))<<1;
   end
   for(genvar i=0;i<REST;i++) assign reduced[2*GROUPS+i]=terms[3*GROUPS+i];
   SimdCarrySavePair #(.B(B),.N(NEXT)) next_level(.terms(reduced),.term0(term0),.term1(term1));
 end
endmodule
module SimdThreeTermSum #(parameter int W=163)(
 input wire [W-1:0] a,b,c,input wire cin,output wire [W:0] result);
 (* keep *) wire [W-1:0] sum0,carry0;
 assign sum0=a^b^c;
 assign carry0=((a&b)|(a&c)|(b&c))<<1;
 SimdPrefixAdd #(.W(W)) final_sum(.a(sum0),.b(carry0),.cin(cin),.result(result));
endmodule
'''

BOOTH_COMPRESSOR = '''
// Radix-4 recoding halves the partial-product rows. Negative rows use
// ones-complement plus one sparse correction vector, avoiding a carry chain
// inside each recoder. All operations are modulo the exact 2*W product width.
module SimdBoothUnsignedMultiply #(parameter int W=53)(
 input wire [W-1:0] a,b,output wire [2*W-1:0] result);
 localparam int B=2*W,N=(W+2)/2;
 wire [2*N:0] multiplier={ {(2*N-W){1'b0}},b,1'b0};
 wire [B-1:0] terms[N+1];
 wire [B-1:0] corrections;
 for(genvar i=0;i<N;i++) begin:recoded
   wire [2:0] code=multiplier[2*i+:3];
   wire negative=code[2];
   wire one=code[0]^code[1];
   wire two=(!code[2] && code[1] && code[0]) ||
            (code[2] && !code[1] && !code[0]);
   wire [W+1:0] magnitude=({2'b0,a}&{(W+2){one}}) |
                         ({1'b0,a,1'b0}&{(W+2){two}});
   wire [W+1:0] encoded=magnitude^{(W+2){negative}};
   assign terms[i]={{(B-W-2){negative}},encoded}<<(2*i);
   assign corrections[2*i]=negative;
   if(2*i+1<B) assign corrections[2*i+1]=1'b0;
 end
 if(2*N<B) assign corrections[B-1:2*N]='0;
 assign terms[N]=corrections;
 SimdCarrySaveReduce #(.B(B),.N(N+1)) tree(.terms(terms),.result(result));
endmodule
'''

BALANCED_PREFIXES = '''
// Keep logarithmic OR-prefix levels: sharing all overlapping reductions can
// otherwise create a measured linear AND/OR chain in the mapped normalizer.
module SimdBalancedPrefixes #(parameter int W=163)(input wire [W-1:0] value,
 output wire [W-1:0] leading,lower);
 localparam int L=$clog2(W);
 (* keep *) wire [W-1:0] up[0:L],down[0:L];
 assign up[0]=value;assign down[0]=value;
 for(genvar level=1;level<=L;level++)begin
   for(genvar j=0;j<W;j++)begin
     if(j+(1<<(level-1))<W)assign up[level][j]=up[level-1][j]|up[level-1][j+(1<<(level-1))];
     else assign up[level][j]=up[level-1][j];
     if(j>=(1<<(level-1)))assign down[level][j]=down[level-1][j]|down[level-1][j-(1<<(level-1))];
     else assign down[level][j]=down[level-1][j];
   end
 end
 assign lower=down[L];
 for(genvar j=0;j<W;j++)begin
   if(j==W-1)assign leading[j]=value[j];
   else assign leading[j]=value[j]&&!up[L][j+1];
 end
endmodule
'''


def split_operation_wrapper():
    text='// SPDX-License-Identifier: Apache-2.0\n// Isolated F2 experiment; both arithmetic cuts are retained.\n'
    for width,format_name,one in[(32,'FP32','3f800000'),(64,'FP64','3ff0000000000000')]:
        text+=f'''module SimdFpF2{width}(input logic clock,ce,input logic [{width-1}:0] a,b,c,
 input logic [2:0] op,output logic [{width-1}:0] result,output logic [4:0] flags);
 logic choose_add_q;
 wire [{width-1}:0] product_result,add_result;
 wire [4:0] product_flags,add_flags;
 wire [{width-1}:0] product_c=(op==0)?{{1'b1,{{{width-1}{{1'b0}}}}}}:c;
 wire [{width-1}:0] add_c=b^((op==3)?{{1'b1,{{{width-1}{{1'b0}}}}}}:{width}'b0);
 always @(posedge clock)if(ce)begin
   choose_add_q<=(op==2 || op==3);
   result<=choose_add_q?add_result:product_result;
   flags<=choose_add_q?add_flags:product_flags;
 end
'''
        for name,operands,result,flags in[
                ('product_core','{product_c,b,a}','product_result','product_flags'),
                ('add_core',"{add_c,a,"+str(width)+"'h"+one+'}','add_result','add_flags')]:
            text+=f''' SimdFpF2Core #(.FpFormat(fpnew_pkg::{format_name}),.NumPipeRegs(1),.PipeConfig(fpnew_pkg::INSIDE)) {name}(
 .clk_i(clock),.rst_ni(1'b1),.operands_i({operands}),.is_boxed_i(3'b111),
 .rnd_mode_i(fpnew_pkg::RNE),.op_i(fpnew_pkg::FMADD),.op_mod_i(1'b0),
 .tag_i(1'b0),.mask_i(1'b1),.aux_i(1'b0),.in_valid_i(ce),.in_ready_o(),
 .flush_i(1'b0),.result_o({result}),.status_o({flags}),.extension_bit_o(),.tag_o(),.mask_o(),.aux_o(),
 .out_valid_o(),.out_ready_i(ce),.busy_o(),.reg_ena_i(ce),.early_out_valid_o());
'''
        text+='endmodule\n'
    return text


def retime_before_sum(text):
    """Move the existing middle cut before addition; keep exactly one cut."""
    replacements=[
        ('  logic                  [0:NUM_MID_REGS][3*PRECISION_BITS+3:0]   mid_pipe_sum_q;\n'
         '  logic                  [0:NUM_MID_REGS]                         mid_pipe_final_sign_q;',
         '  logic [0:NUM_MID_REGS][3*PRECISION_BITS+3:0] mid_pipe_product_q,mid_pipe_addend_q;\n'
         '  logic [0:NUM_MID_REGS] mid_pipe_tent_sign_q;'),
        ('  assign mid_pipe_sum_q[0]         = sum;\n  assign mid_pipe_final_sign_q[0]  = final_sign;',
         '  assign mid_pipe_product_q[0] = product_shifted;\n'
         '  assign mid_pipe_addend_q[0] = addend_after_shift;\n'
         '  assign mid_pipe_tent_sign_q[0] = tentative_sign;'),
        ("    `FFL(mid_pipe_sum_q[i+1],         mid_pipe_sum_q[i],         reg_ena, '0)\n"
         "    `FFL(mid_pipe_final_sign_q[i+1],  mid_pipe_final_sign_q[i],  reg_ena, '0)",
         "    `FFL(mid_pipe_product_q[i+1],mid_pipe_product_q[i],reg_ena,'0)\n"
         "    `FFL(mid_pipe_addend_q[i+1],mid_pipe_addend_q[i],reg_ena,'0)\n"
         "    `FFL(mid_pipe_tent_sign_q[i+1],mid_pipe_tent_sign_q[i],reg_ena,'0)"),
        ('  assign sum_q                   = mid_pipe_sum_q[NUM_MID_REGS];',
         '  assign sum_q = sum;'),
        ('  assign final_sign_q            = mid_pipe_final_sign_q[NUM_MID_REGS];',
         '  assign final_sign_q = final_sign;')]
    for old,new in replacements:
        if text.count(old)!=1:raise ValueError('unexpected FMA middle-cut structure: '+old)
        text=text.replace(old,new)
    begin=text.index('  //Mantissa adder (ab+c).');end=text.index('  // ---------------\n  // Internal pipeline',begin)
    block=text[begin:end]
    names=dict(product_shifted='mid_pipe_product_q[NUM_MID_REGS]',
        addend_after_shift='mid_pipe_addend_q[NUM_MID_REGS]',
        addend_shifted='(effective_subtraction_q ? ~mid_pipe_addend_q[NUM_MID_REGS] : mid_pipe_addend_q[NUM_MID_REGS])',
        inject_carry_in='(effective_subtraction_q & ~sticky_before_add_q)',
        effective_subtraction='effective_subtraction_q',tentative_sign='mid_pipe_tent_sign_q[NUM_MID_REGS]')
    block=re.sub(r'\b('+('|'.join(names))+r')\b',lambda m:names[m[0]],block)
    block='''  // Experiment: the existing single middle register now captures both
  // adder operands. The wrapper still supplies the second and final cut.
  initial begin
    assert(NUM_MID_REGS==1 && NUM_INP_REGS==0 && NUM_OUT_REGS==0);
  end
'''+block
    # The native synthesis frontend requires declarations before their use.
    # Place the post-cut adder after the existing pipeline declarations.
    text=text[:begin]+text[end:]
    marker='  // --------------\n  // Normalization'
    if text.count(marker)!=1:raise ValueError('unexpected normalization boundary')
    return text.replace(marker,block+'\n'+marker)


def run(args):
    # The parent experiment must contain its immutable, individually hashed
    # inputs. This avoids silently rebuilding from later active source edits.
    source = args.source.resolve()
    identity = json.loads((source / 'build_identity.json').read_text())
    inputs = source / 'inputs'
    for name, digest in identity['input_manifest'].items():
        if sha(inputs / name) != digest:
            raise ValueError('parent experimental RTL changed: ' + name)
    label=args.variant+('_fast' if args.mapping_profile=='fast' else '')
    out = ROOT / ('.build/physical/asap7_TT_fp64_2_1x1_1ns_'+label)
    if out.exists():
        raise ValueError('choose a clean experiment directory; existing evidence is retained')
    out.mkdir(parents=True)
    shutil.copytree(inputs, out / 'inputs')
    core = out / 'inputs/SimdFpF2Core.sv'
    text = core.read_text()
    if args.variant=='carry_save':
        old = '  assign raw_product = raw_mantissa_a * raw_mantissa_b;'
        if text.count(old) != 1:
            raise ValueError('unexpected parent multiplier')
        text=text.replace(old, '  SimdUnsignedMultiply #(.W(PRECISION_BITS)) multiplier(\n'
            '    .a(raw_mantissa_a),.b(raw_mantissa_b),.result(raw_product));')
        prefix = out / 'inputs/PrefixAdd.sv'
        prefix.write_text(prefix.read_text() + COMPRESSOR)
    elif args.variant=='zero_product':
        old='''  assign exponent_product = (info_a.is_zero || info_b.is_zero)
                            ? 2 - signed'(BIAS) // in case the product is zero, set minimum exp.
                            : signed'(exponent_a + info_a.is_subnormal
                                      + exponent_b + info_b.is_subnormal
                                      - signed'(BIAS));'''
        if text.count(old)!=1:raise ValueError('unexpected exponent calculation')
        text=text.replace(old,'''  // Zero products bypass the datapath below. Remaining finite exponents
  // depend only on encoded exponent fields, avoiding mantissa-zero fanout
  // through alignment and both wide prefix networks.
  assign exponent_product = signed'(exponent_a + (operand_a.exponent == '0)
                                      + exponent_b + (operand_b.exponent == '0)
                                      - signed'(BIAS));''')
        old='''    end
  end

  // ---------------------------
  // Initial exponent data path'''
        if text.count(old)!=1:raise ValueError('unexpected special-case boundary')
        text=text.replace(old,'''    end else if (info_a.is_zero || info_b.is_zero) begin
      result_is_special = 1'b1;
      special_result = operand_c;
      // IEEE exact zero: equal signs retain the sign; opposite signs use
      // negative zero only for round-towards-minus-infinity. Invalid/NaN/Inf
      // handling above takes precedence over this finite fast path.
      if (info_c.is_zero)
        special_result.sign = (tentative_sign == operand_c.sign) ?
          operand_c.sign : (inp_pipe_rnd_mode_q[NUM_INP_REGS] == fpnew_pkg::RDN);
    end
  end

  // ---------------------------
  // Initial exponent data path''')
    elif args.variant=='fused_sum':
        text=text.replace('logic [2*PRECISION_BITS-1:0] raw_product, add_product;',
            'logic [2*PRECISION_BITS-1:0] raw_product0,raw_product1,add_product;')
        old='''  SimdUnsignedMultiply #(.W(PRECISION_BITS)) multiplier(
    .a(raw_mantissa_a),.b(raw_mantissa_b),.result(raw_product));'''
        if text.count(old)!=1:raise ValueError('expected carry-save multiplier parent')
        text=text.replace(old,'''  SimdUnsignedPairMultiply #(.W(PRECISION_BITS)) multiplier(
    .a(raw_mantissa_a),.b(raw_mantissa_b),.term0(raw_product0),.term1(raw_product1));''')
        old='''  assign product = (inp_pipe_op_q[NUM_INP_REGS] == fpnew_pkg::ADD ||
                    inp_pipe_op_q[NUM_INP_REGS] == fpnew_pkg::ADDS) ? add_product : raw_product;'''
        if text.count(old)!=1:raise ValueError('unexpected add-product bypass')
        text=text.replace(old,'''  wire is_add_product=(inp_pipe_op_q[NUM_INP_REGS] == fpnew_pkg::ADD ||
                    inp_pipe_op_q[NUM_INP_REGS] == fpnew_pkg::ADDS);
  wire [3*PRECISION_BITS+3:0] product_term0,product_term1;
  assign product_term0=is_add_product ?
    {{(PRECISION_BITS+2){1'b0}},add_product,2'b00} :
    {{(PRECISION_BITS+2){1'b0}},raw_product0,2'b00};
  assign product_term1=is_add_product ? '0 :
    {{(PRECISION_BITS+2){1'b0}},raw_product1,2'b00};''')
        text=text.replace('  assign product_shifted = product << 2; // constant shift','')
        old='  SimdPrefixAdd #(.W(3*PRECISION_BITS+4)) positive_sum(.a(product_shifted), .b(addend_shifted), .cin(inject_carry_in), .result(sum_pos));'
        if text.count(old)!=1:raise ValueError('unexpected positive mantissa sum')
        text=text.replace(old,'  SimdThreeTermSum #(.W(3*PRECISION_BITS+4)) positive_sum(.a(product_term0),.b(product_term1),.c(addend_shifted),.cin(inject_carry_in),.result(sum_pos));')
        old="  SimdPrefixAdd #(.W(3*PRECISION_BITS+4)) negative_sum(.a(addend_after_shift), .b(~product_shifted), .cin(1'b1), .result(negative_full));"
        if text.count(old)!=1:raise ValueError('unexpected negative mantissa sum')
        text=text.replace(old,'''  // c-(x+y) = c+~x+~y+2 modulo the mantissa width.
  // The negative candidate's carry is discarded by the unchanged selector.
  (* keep *) wire [3*PRECISION_BITS+3:0] neg_sum0,neg_carry0;
  assign neg_sum0=addend_after_shift^~product_term0^~product_term1;
  assign neg_carry0=((addend_after_shift&~product_term0)|
    (addend_after_shift&~product_term1)|(~product_term0&~product_term1))<<1;
  SimdThreeTermSum #(.W(3*PRECISION_BITS+4)) negative_sum(
    .a(neg_sum0),.b(neg_carry0),.c((3*PRECISION_BITS+4)'(2)),.cin(1'b0),.result(negative_full));''')
        prefix=out/'inputs/PrefixAdd.sv';prefix.write_text(prefix.read_text()+FUSED_COMPRESSOR)
    elif args.variant in('retimed_sum','retimed_sum_v2'):
        if 'norm_choice'not in text:raise ValueError('expected verified onehot normalization parent')
        text=retime_before_sum(text)
    elif args.variant=='balanced_norm':
        if 'assign sum_q = sum;'not in text:raise ValueError('expected verified retimed FMA parent')
        replacements=[
            ('  wire [NORM_W-1:0] norm_choice;', '''  wire [NORM_W-1:0] norm_choice;
  wire [NORM_W-1:0] norm_leading,norm_lower;
  SimdBalancedPrefixes #(.W(NORM_W)) norm_prefixes(.value(sum_q),.leading(norm_leading),.lower(norm_lower));
  (* keep *) wire [2*NORM_T-1:1] normal_valid;
  for(genvar k=0;k<NORM_T;k++)begin
    if(k<NORM_W)assign normal_valid[NORM_T+k]=norm_choice[k];
    else assign normal_valid[NORM_T+k]=1'b0;
  end
  for(genvar k=1;k<NORM_T;k++)
    assign normal_valid[k]=normal_valid[2*k]|normal_valid[2*k+1];'''),
            ('''      wire leading;
      if(j==NORM_W-1) assign leading=sum_q[j];
      else assign leading=sum_q[j] && !(|sum_q[NORM_W-1:j+1]);''',
             '      wire leading=norm_leading[j];'),
            ('assign sticky_rest=|sum_q[j-PRECISION_BITS-2:0];',
             'assign sticky_rest=norm_lower[j-PRECISION_BITS-2];'),
            ('  wire [NORM_CW-1:0] norm_tree[1:2*NORM_T-1];',
             '  (* keep *) wire [NORM_CW-1:0] norm_tree[1:2*NORM_T-1];'),
            ('    if(|norm_choice) begin','    if(normal_valid[1]) begin')]
        for old,new in replacements:
            if text.count(old)!=1:raise ValueError('unexpected normalization tree: '+old)
            text=text.replace(old,new)
        prefix=out/'inputs/PrefixAdd.sv';prefix.write_text(prefix.read_text()+BALANCED_PREFIXES)
    elif args.variant=='split_add_prefix':
        old='  wire [NORM_W-1:0] norm_choice;'
        if text.count(old)!=1:raise ValueError('expected onehot normalization parent')
        text=text.replace(old,old+'''
  wire [NORM_W-1:0] norm_leading,norm_lower;
  SimdBalancedPrefixes #(.W(NORM_W)) norm_prefixes(.value(sum_q),.leading(norm_leading),.lower(norm_lower));''')
        old='''      wire leading;
      if(j==NORM_W-1) assign leading=sum_q[j];
      else assign leading=sum_q[j] && !(|sum_q[NORM_W-1:j+1]);'''
        if text.count(old)!=1:raise ValueError('unexpected onehot leading detector')
        text=text.replace(old,'      wire leading=norm_leading[j];')
        old='assign sticky_rest=|sum_q[j-PRECISION_BITS-2:0];'
        if text.count(old)!=1:raise ValueError('unexpected normalization sticky reduction')
        text=text.replace(old,'assign sticky_rest=norm_lower[j-PRECISION_BITS-2];')
        prefix=out/'inputs/PrefixAdd.sv';prefix.write_text(prefix.read_text()+BALANCED_PREFIXES)
        (out/'inputs/SimdFpF2.sv').write_text(split_operation_wrapper())
    elif args.variant=='booth':
        old='  SimdUnsignedMultiply #(.W(PRECISION_BITS)) multiplier('
        if text.count(old)!=1:raise ValueError('expected unsigned multiplier parent')
        text=text.replace(old,'  SimdBoothUnsignedMultiply #(.W(PRECISION_BITS)) multiplier(')
        prefix=out/'inputs/PrefixAdd.sv'
        prefix.write_text(prefix.read_text()+BOOTH_COMPRESSOR)
    elif args.variant=='prefix_pruned':
        prefix=out/'inputs/PrefixAdd.sv'
        old=' (* keep *) wire [W-1:0] p[0:L],g[0:L];'
        raw=prefix.read_text()
        if raw.count(old)!=1:raise ValueError('unexpected prefix retention attributes')
        prefix.write_text(raw.replace(old,' wire [W-1:0] p[0:L],g[0:L];'))
    else:
        begin=text.index('  assign sum_lower = sum_q[LOWER_SUM_WIDTH-1:0];')
        end=text.index('  // Update the sticky bit with the shifted-out bits',begin)
        text=text[:begin]+ONEHOT_NORMALIZATION+'\n'+text[end:]
        text,n=re.subn(r'(assign mid_pipe_exp_prod_q\[0\]\s*=\s*)exponent_product;',
            r"\1(exponent_difference > signed'(PRECISION_BITS+2)) ? tentative_exponent-(3*PRECISION_BITS+3) : exponent_product-2*PRECISION_BITS;",text)
        if n!=1:raise ValueError('normalization anchor register not found')
        text,n=re.subn(r'(assign mid_pipe_exp_diff_q\[0\]\s*=\s*)exponent_difference;',
            r"\1(exponent_difference > signed'(PRECISION_BITS+2)) ? tentative_exponent-1 : exponent_product+PRECISION_BITS+2;",text)
        if n!=1:raise ValueError('minimum-exponent shift register not found')
    core.write_text(text)
    paths = [out / 'inputs' / name for name in identity['input_manifest']]
    identity.update(rtl_sha256=rtl_digest(paths, out / 'inputs/common_cells'),
        input_manifest={p.name: sha(p) for p in paths},
        experiment=args.variant,
        provenance_status='experimental_component_only',
        experiment_script_sha256=sha(__file__),
        parent_identity_sha256=sha(source / 'build_identity.json'))
    (out / 'build_identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    shutil.copy2(__file__,out/'experiment_driver.py')
    wrapper = out / 'GateDut.sv'
    wrapper.write_text('module GateDut(input clock,ce,input [63:0]a,b,c,input [2:0]op,'
        'output [63:0]result,output [4:0]flags);SimdFpF264 u(.*);endmodule\n')
    with (out / 'rtl_build.log').open('w') as log:
        subprocess.run(['verilator', '--cc', '--exe', '--build', '-j', str(args.jobs), '-Wno-fatal',
            '--top-module', 'GateDut', '--Mdir', str(out/'rtl_obj'),
            '-I'+str(out/'inputs'), *map(str, paths), str(wrapper),
            str(ROOT/'tests/scalar_gate.cpp'), '-o', str(out/'rtl_sim')],
            check=True, stdout=log, stderr=subprocess.STDOUT)
    rng = random.Random(981)
    edges = [0, 1, 0x8000000000000000, 0x7ff0000000000000, 0xfff0000000000000,
        0x7ff8000000000001, 0x7ff0000000000001, 0xfffffffffffff,
        0x10000000000000, 0x7fefffffffffffff,0x3ff0000000000000,
        0xbff0000000000000,0x3ff0000000000001]
    vectors = []
    for op in range(4):
        for i in range(args.count):
            a,b,c = [rng.getrandbits(64) for _ in range(3)]
            if i < len(edges)**2:
                a,b = edges[i%len(edges)], edges[i//len(edges)]
            value,flags = fp_operation(a,b,c,Format.F64,Op(op))
            vectors.append(f'{a:x} {b:x} {c:x} {op} {value:x} {flags}\n')
        for a in edges:
            for b in edges:
                for c in edges:
                    value,flags=fp_operation(a,b,c,Format.F64,Op(op))
                    vectors.append(f'{a:x} {b:x} {c:x} {op} {value:x} {flags}\n')
    for i in range(args.cancellation_count):
        offset=rng.randrange(-500,501)
        a=(rng.getrandbits(64)&0x800fffffffffffff)|((1023+offset)<<52)
        b=(rng.getrandbits(64)&0x800fffffffffffff)|((1023-offset+rng.randrange(-2,3))<<52)
        product,_=fp_operation(a,b,0,Format.F64,Op(0));c=product^0x8000000000000000
        value,flags=fp_operation(a,b,c,Format.F64,Op(1))
        vectors.append(f'{a:x} {b:x} {c:x} 1 {value:x} {flags}\n')
    trace = out / 'rtl_vectors.txt'
    trace.write_text(''.join(vectors))
    result = subprocess.run([str(out/'rtl_sim'), str(trace), '2', 'elastic'],
        check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    report.update(rtl_sha256=identity['rtl_sha256'], trace_sha256=sha(trace),
        simulator_sha256=sha(out/'rtl_sim'), testbench_sha256=sha(ROOT/'tests/scalar_gate.cpp'),
        reference_sha256=sha(ROOT/'python/simd_eval/reference.py'), pipeline_stages=2,
        main_design_changed=False, scope='FP64 only', physical_frequency_certified=False)
    (out/'rtl_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print('RTL verified:', json.dumps(report), flush=True)
    env = os.environ.copy()
    local = ROOT/'.deps/tools/openroad'
    env['LD_LIBRARY_PATH'] = ':'.join([str(local/'opt/or-tools/lib'),
        str(local/'usr/lib/x86_64-linux-gnu'),env.get('LD_LIBRARY_PATH','')])
    synthesize(out, identity['top'], paths, profile('TT')['liberty'], 1.0, env,
        mapping_profile=args.mapping_profile)
    identity = json.loads((out/'build_identity.json').read_text())
    identity['mapped_sha256'] = sha(out/'synth.v')
    (out/'build_identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    subprocess.run([sys.executable, str(ROOT/'physical/run.py'), '--target','fp64',
        '--stages','2','--rows','1','--period','1','--label',label,
        '--reuse-mapped','--cts-only'], check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--count', type=int, default=1000, help='vectors per operation')
    p.add_argument('--variant',choices=['carry_save','zero_product','prefix_pruned','onehot_norm','fused_sum','booth','split_add_prefix','retimed_sum','retimed_sum_v2','balanced_norm'],default='carry_save')
    p.add_argument('--cancellation-count',type=int,default=0)
    p.add_argument('--mapping-profile',choices=['fast','full'],default='fast')
    p.add_argument('-j','--jobs', type=int, default=8)
    run(p.parse_args())
