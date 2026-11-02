"""Independent carry-save exponent calculations inside the existing F2 stage."""
import json
import re
import shutil
import subprocess
from pathlib import Path

from run import sha

RTL='''
// No registers. All sums are modular in the original signed exponent width.
module SimdParallelExponents #(parameter E=13,B=1023)(
 input wire [E-1:0] a,b,c,input wire za,zb,zc,
 output wire [E-1:0] product,difference);
 localparam [E-1:0] PRODUCT_BIAS=-B,DIFFERENCE_BIAS=B+2;
 wire [1:0] product_correction={za&zb,za^zb};
 wire [1:0] difference_correction={(za&zb)|((~zc)&(za|zb)),zc^za^zb};
 wire [4*E-1:0] product_terms;
 wire [5*E-1:0] difference_terms;
 assign product_terms={{{(E-2){1'b0}},product_correction},PRODUCT_BIAS,b,a};
 assign difference_terms={{{(E-2){difference_correction[1]}},difference_correction},
                         DIFFERENCE_BIAS,~b,~a,c};
 SimdExponentReduce #(.E(E),.N(4)) p(.terms(product_terms),.result(product));
 SimdExponentReduce #(.E(E),.N(5)) d(.terms(difference_terms),.result(difference));
endmodule
module SimdExponentReduce #(parameter E=13,N=4)(
 input wire [E*N-1:0] terms,output wire [E-1:0] result);
 if(N==2)begin
   wire [E:0] full_sum;
   SimdPrefixAdd #(.W(E)) finish(.a(terms[E-1:0]),.b(terms[2*E-1:E]),.cin(1'b0),.result(full_sum));
   assign result=full_sum[E-1:0];
 end else begin
   localparam G=N/3,REST=N%3,NEXT=2*G+REST;
   wire [E*NEXT-1:0] reduced;
   for(genvar i=0;i<G;i++)begin:compress
     wire [E-1:0] a=terms[E*(3*i)+:E],b=terms[E*(3*i+1)+:E],c=terms[E*(3*i+2)+:E];
     assign reduced[E*(2*i)+:E]=a^b^c;
     assign reduced[E*(2*i+1)+:E]=((a&b)|(a&c)|(b&c))<<1;
   end
   for(genvar i=0;i<REST;i++)assign reduced[E*(2*G+i)+:E]=terms[E*(3*G+i)+:E];
   SimdExponentReduce #(.E(E),.N(NEXT)) next_level(.terms(reduced),.result(result));
 end
endmodule
'''

EFFECTIVE_RTL='''
// za/zb are the exact zero predicates of the encoded, padded exponents.
module SimdParallelExponents #(parameter E=13,B=1023)(
 input wire [E-1:0] a,b,c,input wire za,zb,zc,
 output wire [E-1:0] product,difference);
 localparam [E-1:0] PRODUCT_BIAS=-B,DIFFERENCE_BIAS=B+2;
 wire [E-1:0] effective_a=a|{{(E-1){1'b0}},za};
 wire [E-1:0] effective_b=b|{{(E-1){1'b0}},zb};
 wire [E:0] effective_c;
 SimdPrefixAdd #(.W(E)) addend(.a(c),.b({E{1'b0}}),.cin(zc),.result(effective_c));
 wire [3*E-1:0] product_terms={PRODUCT_BIAS,effective_b,effective_a};
 wire [4*E-1:0] difference_terms={DIFFERENCE_BIAS,~effective_b,~effective_a,effective_c[E-1:0]};
 SimdExponentReduce #(.E(E),.N(3)) p(.terms(product_terms),.result(product));
 SimdExponentReduce #(.E(E),.N(4)) d(.terms(difference_terms),.result(difference));
endmodule
'''+RTL[RTL.index('module SimdExponentReduce '):]


def transform(text):
    product=re.search(r'  assign exponent_product\s*=\s*.*?;',text,flags=re.S)
    if not product or not all(s in product[0]for s in("operand_a.exponent == '0","operand_b.exponent == '0","signed'(BIAS)")):
        raise ValueError('parallel exponent candidate requires the verified encoded-zero correction parent')
    text,n=re.subn(r'  assign exponent_product\s*=\s*.*?;', '''  SimdParallelExponents #(.E(EXP_WIDTH),.B(BIAS)) parallel_exponents(
    .a(exponent_a),.b(exponent_b),.c(exponent_c),
    .za(operand_a.exponent=='0),.zb(operand_b.exponent=='0),.zc(~info_c.is_normal),
    .product(exponent_product),.difference(exponent_difference));''',text,flags=re.S)
    if n!=1:raise ValueError('expected one initial product exponent assignment')
    text,n=re.subn(r'  assign exponent_difference\s*=\s*exponent_addend - exponent_product;',
        '  // Exponent difference is independently calculated above.',text)
    if n!=1:raise ValueError('expected one initial exponent difference assignment')
    return text


def prove(prefix,out,effective=False):
    """SAT compares every input bit and correction flag at four actual widths."""
    out=Path(out);out.mkdir();prefix=Path(prefix)
    shutil.copy2(__file__,out/'proof_driver.py')
    actual_prefix=prefix.read_text().split('endmodule',1)[0]+'endmodule\n'
    rtl=out/'exponents.sv';rtl.write_text(actual_prefix+(EFFECTIVE_RTL if effective else RTL))
    checks=[]
    for width,bias in[(6,7),(7,15),(10,127),(13,1023)]:
        name=f'e{width}_b{bias}';miter=out/(name+'.sv');script=out/(name+'.ys');log=out/(name+'.log')
        flags="input zc,"if effective else"input za,zb,zc,"
        predicates="wire za=(a=='0),zb=(b=='0);"if effective else""
        miter.write_text(f'''module ExponentMiter(input [{width-1}:0] a,b,c,{flags}output equivalent);
{predicates}
wire [{width-1}:0] p,d;
SimdParallelExponents #(.E({width}),.B({bias})) dut(.*,.product(p),.difference(d));
wire [{width-1}:0] reference_p=a+za+b+zb-{width}'d{bias};
wire [{width-1}:0] reference_d=c+zc-reference_p;
assign equivalent=(p==reference_p)&&(d==reference_d);
endmodule
''')
        script.write_text(f'read_verilog -sv {rtl} {miter}\nhierarchy -check -top ExponentMiter\nproc\nflatten\nopt\nsat -prove equivalent 1 -verify -timeout 90\n')
        with log.open('w')as stream:
            p=subprocess.run(['yosys','-Q','-T','-s',str(script)],stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        check=dict(width=width,bias=bias,passed=p.returncode==0,log_sha256=sha(log),script_sha256=sha(script))
        checks.append(check)
        if p.returncode:raise RuntimeError('parallel exponent SAT failed or timed out: '+name)
    record=dict(passed=True,method='SAT for all modular exponent inputs; A/B zero flags derived; arbitrary C correction'if effective else'SAT for all modular exponent and correction-bit inputs',
        checks=checks,primitive_sha256=sha(rtl),source_prefix_sha256=sha(prefix),
        driver_sha256=sha(__file__),retiming_performed=False,physical_frequency_certified=False)
    (out/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
