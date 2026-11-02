"""Move two product compressors across the existing F2 register boundary.

Four signed, modular carry-save terms cross the first boundary; the second
stage compresses them and optionally the accumulator before the carry adder.
No register boundary is added. Used only by isolated experiment copies.
"""


def rtl():
    from signed_product_pair import RTL
    core=RTL[:RTL.index('module SimdF2PairReduce ')]
    core=core.replace('output wire [R-1:0] term0,term1);',
        'output wire [R-1:0] term0,term1,term2,term3);')
    core=core.replace('SimdF2PairReduce #','SimdF2FourTermReduce #').replace(
        '.term0(term0),.term1(term1));','.term0(term0),.term1(term1),.term2(term2),.term3(term3));')
    core+='''
module SimdF2FourTermReduce #(parameter R=145,N=33)(
 input wire [R*N-1:0] terms,output wire [R-1:0] term0,term1,term2,term3);
 if(N<=4)begin
   wire [4*R-1:0] padded={{((4-N)*R){1'b0}},terms};
   assign {term3,term2,term1,term0}=padded;
 end else begin
   localparam G=N/3,REST=N%3,NEXT=2*G+REST;
   wire [R*NEXT-1:0] reduced;
   for(genvar i=0;i<G;i++)begin:compress
     wire [R-1:0] a=terms[R*(3*i)+:R],b=terms[R*(3*i+1)+:R],c=terms[R*(3*i+2)+:R];
     assign reduced[R*(2*i)+:R]=a^b^c;
     assign reduced[R*(2*i+1)+:R]=((a&b)|(a&c)|(b&c))<<1;
   end
   for(genvar i=0;i<REST;i++)assign reduced[R*(2*G+i)+:R]=terms[R*(3*G+i)+:R];
   SimdF2FourTermReduce #(.R(R),.N(NEXT)) next_level(.terms(reduced),
     .term0(term0),.term1(term1),.term2(term2),.term3(term3));
 end
endmodule
'''
    for width,result_width in((4,33),(8,33),(16,49),(32,81),(64,145)):
        core+=f'''
module SimdF2SignedProduct{width}x{result_width}(input wire [{width-1}:0] a,b,
 output wire [{result_width-1}:0] term0,term1,term2,term3);
 SimdF2SignedBoothPair #(.W({width}),.R({result_width})) core(
   .a(a),.b(b),.term0(term0),.term1(term1),.term2(term2),.term3(term3));
endmodule
'''
    return core


def transform(text):
    changes=[
        ('val term0=Output(UInt(r.W));val term1=Output(UInt(r.W))',
         'val term0=Output(UInt(r.W));val term1=Output(UInt(r.W))\n'
         '    val term2=Output(UInt(r.W));val term3=Output(UInt(r.W))'),
        ('      val pairRight=RegEnable(partials.io.term1,ce)',
         '''      val pairRight=RegEnable(partials.io.term1,ce)
      val pairThird=RegEnable(partials.io.term2,ce)
      val pairFourth=RegEnable(partials.io.term3,ce)'''),
        ('      val macSum=pairLeft^pairRight^oldWide\n'
         '      val macCarry=(((pairLeft&pairRight)|(pairLeft&oldWide)|(pairRight&oldWide))<<1)(aw,0)',
         '''      val productSum=pairLeft^pairRight^pairThird
      val productCarry=(((pairLeft&pairRight)|(pairLeft&pairThird)|(pairRight&pairThird))<<1)(aw,0)
      val reducedSum=productSum^productCarry^pairFourth
      val reducedCarry=(((productSum&productCarry)|(productSum&pairFourth)|(productCarry&pairFourth))<<1)(aw,0)
      val macSum=reducedSum^reducedCarry^oldWide
      val macCarry=(((reducedSum&reducedCarry)|(reducedSum&oldWide)|(reducedCarry&oldWide))<<1)(aw,0)'''),
        ('FastCarry.add(pairLeft,pairRight,aw+1).asSInt',
         'FastCarry.add(reducedSum,reducedCarry,aw+1).asSInt')]
    for old,new in changes:
        if text.count(old)!=1:raise ValueError('unexpected two-term product boundary: '+old[:70])
        text=text.replace(old,new)
    return text
