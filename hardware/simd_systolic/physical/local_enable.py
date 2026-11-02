"""Replicate the existing two valid bits beside F2 register banks."""
SCALA='''class F2LocalAdvance extends BlackBox {
  override def desiredName="SimdF2LocalAdvance"
  val io=IO(new Bundle {val clock=Input(Clock());val reset=Input(Bool())
    val inValid=Input(Bool());val outReady=Input(Bool());val ce=Output(Bool())})
}
'''
RTL='''
// Exact local copy of F2 admission/idle/stall control. Arithmetic stays F2.
(* keep_hierarchy=1 *)
module SimdF2LocalAdvance(input wire clock,reset,inValid,outReady,output wire ce);
 reg first_valid,second_valid;
 wire advance=!second_valid||outReady;
 assign ce=advance&&(inValid||first_valid||second_valid);
 always @(posedge clock)begin
   if(reset)begin first_valid<=1'b0;second_valid<=1'b0;end
   else if(ce)begin first_valid<=inValid&&advance;second_valid<=first_valid;end
 end
endmodule
'''


def transform(text):
    if text.count('class SimdAlu(')!=1:raise ValueError('unexpected ALU declaration')
    text=text.replace('class SimdAlu(',SCALA+'\nclass SimdAlu(',1)
    anchor='  val q=Pipes.delay(io.in.bits,cfg.stages,ce)'
    if text.count(anchor)!=1:raise ValueError('unexpected request metadata delay')
    text=text.replace(anchor,'''  def localAdvance():Bool = {
    val control=Module(new F2LocalAdvance)
    control.io.clock:=clock;control.io.reset:=reset.asBool
    control.io.inValid:=io.in.valid;control.io.outReady:=io.out.ready
    assert(control.io.ce===ce,"local F2 advance diverged")
    control.io.ce
  }
  val requestWidth=(new Request).getWidth
  val requestWords=io.in.bits.asUInt.pad(((requestWidth+31)/32)*32)
  val q=Cat((0 until (requestWidth+31)/32).reverse.map {word=>
    Pipes.delay(requestWords(32*word+31,32*word),cfg.stages,localAdvance())
  })(requestWidth-1,0).asTypeOf(new Request)''')
    replacements=[
        ('else (0 until 64/w).map {lane=>\n      val (data,flags)=IntegerToFloat',
         'else (0 until 64/w).map {lane=>\n      val ce=localAdvance()\n      val (data,flags)=IntegerToFloat'),
        ('    val pieces=(0 until 64/w).map { lane=>',
         '    val pieces=(0 until 64/w).map { lane=>\n      val ce=localAdvance()'),
        ('    u.io.clock:=clock;u.io.ce:=ce','    u.io.clock:=clock;u.io.ce:=localAdvance()'),
        ('  d.io.clock:=clock;d.io.ce:=ce;','  d.io.clock:=clock;d.io.ce:=localAdvance();')]
    for old,new in replacements:
        if text.count(old)!=1:raise ValueError('unexpected local enable site: '+old)
        text=text.replace(old,new)
    return text
