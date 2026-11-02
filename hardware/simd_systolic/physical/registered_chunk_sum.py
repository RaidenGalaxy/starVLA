"""One-cut three-row sum: CSA and local candidates before group carry."""
RTL='''
module SimdRegisteredChunkSum #(parameter W=164,B=8)(
 input wire clock,ce,input wire [W-1:0] a,b,c,input wire [1:0] bias,
 output wire [W-1:0] result);
 localparam N=(W+B-1)/B,L=$clog2(N);
 wire [W-1:0] first=a^b^c,first_carry=((a&b)|(a&c)|(b&c))<<1;
 wire [W-1:0] extra={{(W-2){1'b0}},bias};
 wire [W-1:0] left=first^first_carry^extra;
 wire [W-1:0] right=((first&first_carry)|(first&extra)|(first_carry&extra))<<1;
 reg [B-1:0] low0[N],low1[N];
 reg [N-1:0] generate_q,propagate_q;
 (* keep *) wire [N-1:0] g[0:L],p[0:L];
 wire [N:0] carry;
 assign g[0]=generate_q;assign p[0]=propagate_q;assign carry[0]=1'b0;
 for(genvar k=0;k<N;k++)begin:chunk
   localparam C=(W-k*B<B)?W-k*B:B;
   wire [C:0] sum0={1'b0,left[k*B+:C]}+{1'b0,right[k*B+:C]};
   wire [C:0] sum1=sum0+1'b1;
   always @(posedge clock)if(ce)begin
     low0[k]<=sum0[C-1:0];low1[k]<=sum1[C-1:0];
     generate_q[k]<=sum0[C];propagate_q[k]<=sum1[C];
   end
   assign carry[k+1]=g[L][k];
   assign result[k*B+:C]=carry[k]?low1[k][C-1:0]:low0[k][C-1:0];
 end
 for(genvar level=1;level<=L;level++)begin:prefix
   for(genvar k=0;k<N;k++)begin
     if(k>=(1<<(level-1)))begin
       assign g[level][k]=g[level-1][k]|(p[level-1][k]&g[level-1][k-(1<<(level-1))]);
       assign p[level][k]=p[level-1][k]&p[level-1][k-(1<<(level-1))];
     end else begin
       assign g[level][k]=g[level-1][k];assign p[level][k]=p[level-1][k];
     end
   end
 end
endmodule
'''


def retime(text):
    positive='  SimdThreeTermSum #(.W(3*PRECISION_BITS+4)) positive_sum(.a(product_term0),.b(product_term1),.c(addend_shifted),.cin(inject_carry_in),.result(sum_pos));'
    if text.count(positive)!=1:raise ValueError('expected verified carry-save product parent')
    text=text.replace(positive,'''  SimdRegisteredChunkSum #(.W(3*PRECISION_BITS+5),.B(8)) positive_sum(
    .clock(clk_i),.ce(reg_ena_i[0]),.a({1'b0,product_term0}),.b({1'b0,product_term1}),
    .c({1'b0,addend_shifted}),.bias({1'b0,inject_carry_in}),.result(sum_pos));''')
    begin=text.index('  (* keep *) wire [3*PRECISION_BITS+3:0] neg_sum0,neg_carry0;')
    end=text.index('  assign sum_neg =',begin)
    text=text[:begin]+'''  SimdRegisteredChunkSum #(.W(3*PRECISION_BITS+5),.B(8)) negative_sum(
    .clock(clk_i),.ce(reg_ena_i[0]),.a({1'b0,addend_after_shift}),
    .b({1'b0,~product_term0}),.c({1'b0,~product_term1}),.bias(2'd2),.result(negative_full));
'''+text[end:]
    replacements=[
        ('  logic                  [0:NUM_MID_REGS][3*PRECISION_BITS+3:0]   mid_pipe_sum_q;\n',''),
        ('  assign mid_pipe_sum_q[0]         = sum;\n',''),
        ("    `FFL(mid_pipe_sum_q[i+1],         mid_pipe_sum_q[i],         reg_ena, '0)\n",''),
        ('  assign mid_pipe_final_sign_q[0]  = final_sign;',
         '  assign mid_pipe_final_sign_q[0]  = tentative_sign;'),
        ('  assign sum_q                   = mid_pipe_sum_q[NUM_MID_REGS];','  assign sum_q = sum;'),
        ('  assign final_sign_q            = mid_pipe_final_sign_q[NUM_MID_REGS];','  assign final_sign_q = final_sign;')]
    for old,new in replacements:
        if text.count(old)!=1:raise ValueError('unexpected FMA register cut: '+old)
        text=text.replace(old,new)
    text=text.replace('mid_pipe_final_sign_q','mid_pipe_tent_sign_q')
    begin=text.index('  //Mantissa adder (ab+c).');end=text.index('  // ---------------\n  // Internal pipeline',begin)
    block=text[begin:end];split=block.index('  assign sum        =')
    selection=block[split:].replace('effective_subtraction','effective_subtraction_q').replace(
        'tentative_sign','mid_pipe_tent_sign_q[NUM_MID_REGS]')
    block=block[:split]+selection
    text=text[:begin]+text[end:]
    marker='  // --------------\n  // Normalization'
    if text.count(marker)!=1:raise ValueError('unexpected FMA normalization boundary')
    return text.replace(marker,block+'\n'+marker)


def predecoded_rtl():
    """Precompute each local candidate's prefix flags at the same cut.

    Leading/lower flags describe the low W-1 bits, excluding the FMA carry.
    The candidate prefix networks after the cut operate on chunk flags only.
    """
    text=RTL.replace('output wire [W-1:0] result);',
        'output wire [W-1:0] result,output wire [W-2:0] leading,lower);')
    text=text.replace(' wire [N:0] carry;', ''' wire [N:0] carry;
 reg [B-1:0] lead0[N],lead1[N],lower0[N],lower1[N];
 (* keep *) wire [N-1:0] up[0:L],down[0:L];''')
    marker='   assign result[k*B+:C]=carry[k]?low1[k][C-1:0]:low0[k][C-1:0];'
    text=text.replace(marker,marker+'''
   localparam D=(W-1-k*B<B)?W-1-k*B:B;
   if(D>0)begin
     wire [B-1:0] local_lead=carry[k]?lead1[k]:lead0[k];
     wire [B-1:0] local_lower=carry[k]?lower1[k]:lower0[k];
     assign up[0][k]=local_lower[D-1];assign down[0][k]=local_lower[D-1];
     for(genvar j=0;j<D;j++)begin:flag
       if(j==D-1)begin
         always @(posedge clock)if(ce)begin lead0[k][j]<=sum0[j];lead1[k][j]<=sum1[j];end
       end else begin
         always @(posedge clock)if(ce)begin
           lead0[k][j]<=sum0[j]&&!(|sum0[D-1:j+1]);
           lead1[k][j]<=sum1[j]&&!(|sum1[D-1:j+1]);
         end
       end
       always @(posedge clock)if(ce)begin
         lower0[k][j]<=|sum0[j:0];lower1[k][j]<=|sum1[j:0];
       end
       if(k==N-1)assign leading[k*B+j]=local_lead[j];
       else assign leading[k*B+j]=local_lead[j]&&!up[L][k+1];
       if(k==0)assign lower[k*B+j]=local_lower[j];
       else assign lower[k*B+j]=local_lower[j]||down[L][k-1];
     end
   end else begin
     assign up[0][k]=1'b0;assign down[0][k]=1'b0;
   end''')
    marker='   for(genvar k=0;k<N;k++)begin\n'
    text=text.replace(marker,marker+'''
     if(k+(1<<(level-1))<N)
       assign up[level][k]=up[level-1][k]|up[level-1][k+(1<<(level-1))];
     else assign up[level][k]=up[level-1][k];
     if(k>=(1<<(level-1)))
       assign down[level][k]=down[level-1][k]|down[level-1][k-(1<<(level-1))];
     else assign down[level][k]=down[level-1][k];
''')
    return text
