"""Signed Booth product retained as two modular carry-save terms."""
SCALA='''class F2SignedProductPair(w:Int,r:Int) extends BlackBox {
  override def desiredName=s"SimdF2SignedProduct${w}x$r"
  val io=IO(new Bundle {val a=Input(UInt(w.W));val b=Input(UInt(w.W))
    val term0=Output(UInt(r.W));val term1=Output(UInt(r.W))})
}
'''

RTL='''
// Combinational signed product modulo2^R. No register is added here.
module SimdF2SignedBoothPair #(parameter W=64,R=145)(
 input wire [W-1:0] a,b,output wire [R-1:0] term0,term1);
 localparam N=(W+1)/2;
 wire [2*N:0] digits={{(2*N-W){b[W-1]}},b,1'b0};
 wire [R*(N+1)-1:0] terms;
 wire [N-1:0] negative;
 for(genvar i=0;i<N;i++)begin:row
   wire [2:0] code=digits[2*i+:3];
   wire one=(code==1)||(code==2)||(code==5)||(code==6);
   wire two=(code==3)||(code==4);
   wire [W+1:0] extended={{2{a[W-1]}},a};
   wire [W+1:0] twice={extended[W:0],1'b0};
   wire [W+1:0] magnitude=(extended&{(W+2){one}})|(twice&{(W+2){two}});
   wire [W+1:0] encoded=magnitude^{(W+2){code[2]}};
   assign negative[i]=code[2];
   assign terms[R*i+:R]={{(R-W-2){encoded[W+1]}},encoded}<<(2*i);
 end
 for(genvar bit_index=0;bit_index<R;bit_index++)begin:correction
   if(bit_index%2==0 && bit_index/2<N)
     assign terms[R*N+bit_index]=negative[bit_index/2];
   else assign terms[R*N+bit_index]=1'b0;
 end
 SimdF2PairReduce #(.R(R),.N(N+1)) reduce(.terms(terms),.term0(term0),.term1(term1));
endmodule
module SimdF2PairReduce #(parameter R=145,N=33)(
 input wire [R*N-1:0] terms,output wire [R-1:0] term0,term1);
 if(N==1)begin
   assign term0=terms[R-1:0];assign term1={R{1'b0}};
 end else if(N==2)begin
   assign term0=terms[R-1:0];assign term1=terms[2*R-1:R];
 end else begin
   localparam G=N/3,REST=N%3,NEXT=2*G+REST;
   wire [R*NEXT-1:0] reduced;
   for(genvar i=0;i<G;i++)begin:compress
     wire [R-1:0] a=terms[R*(3*i)+:R],b=terms[R*(3*i+1)+:R],c=terms[R*(3*i+2)+:R];
     assign reduced[R*(2*i)+:R]=a^b^c;
     assign reduced[R*(2*i+1)+:R]=((a&b)|(a&c)|(b&c))<<1;
   end
   for(genvar i=0;i<REST;i++)assign reduced[R*(2*G+i)+:R]=terms[R*(3*G+i)+:R];
   SimdF2PairReduce #(.R(R),.N(NEXT)) next_level(.terms(reduced),.term0(term0),.term1(term1));
 end
endmodule
'''

# Fixed native interfaces let Slang import the primitive without parameter
# overrides; the native Verilog frontend specializes the internal core.
for width,result_width in((4,33),(8,33),(16,49),(32,81),(64,145)):
    RTL+=f'''
module SimdF2SignedProduct{width}x{result_width}(input wire [{width-1}:0] a,b,
 output wire [{result_width-1}:0] term0,term1);
 SimdF2SignedBoothPair #(.W({width}),.R({result_width})) core(
   .a(a),.b(b),.term0(term0),.term1(term1));
endmodule
'''


def baugh_rtl():
    """Two signed Baugh-Wooley terms, with sparse constant corrections.

    Complement the two negative sign cross-products. Their constant bias is
    2**W - 2**(2*W-1), represented modulo2**R before the carry-save tree.
    """
    begin=RTL.index('module SimdF2SignedBoothPair ')
    end=RTL.index('module SimdF2PairReduce ',begin)
    body='''module SimdF2SignedBaughPair #(parameter W=64,R=145)(
 input wire [W-1:0] a,b,output wire [R-1:0] term0,term1);
 wire [R*(W+1)-1:0] terms;
 for(genvar i=0;i<W;i++)begin:row
   for(genvar k=0;k<R;k++)begin:column
     if(k>=i && k<i+W)begin
       if((i==W-1)!=(k-i==W-1))assign terms[R*i+k]=~(a[k-i]&b[i]);
       else assign terms[R*i+k]=a[k-i]&b[i];
     end else assign terms[R*i+k]=1'b0;
   end
 end
 for(genvar k=0;k<R;k++)begin:bias
   assign terms[R*W+k]=(k==W)||(k>=2*W-1);
 end
 SimdF2PairReduce #(.R(R),.N(W+1)) reduce(.terms(terms),.term0(term0),.term1(term1));
endmodule
'''
    return(RTL[:begin]+body+RTL[end:]).replace('SimdF2SignedBoothPair #','SimdF2SignedBaughPair #')
