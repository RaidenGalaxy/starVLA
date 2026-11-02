"""Keep exact FP opcode normalization local to each floating datapath."""
SCALA='''class F2FpOpcodeLocalizer extends BlackBox {
  override def desiredName="SimdF2FpOpcodeLocalizer"
  val io=IO(new Bundle {val op=Input(UInt(3.W));val value=Output(UInt(2.W))})
}
'''
RTL='''
// Scalar FP opcodes4..7 have always selected MAC. Normalize locally, so
// the wide data paths do not share one high-fanout opcode decoder.
(* keep=1,keep_hierarchy=1 *)
module SimdF2FpOpcodeLocalizer(input wire [2:0] op,output wire [1:0] value);
 assign value={~op[2]&op[1],op[2]|op[0]};
endmodule
'''


def transform(text):
    if text.count('class SimdAlu(')!=1:raise ValueError('unexpected ALU declaration')
    text=text.replace('class SimdAlu(',SCALA+'\nclass SimdAlu(',1)
    marker='  val fps=(0 until 16).map { lane=>'
    if text.count(marker)!=1:raise ValueError('unexpected FP lane construction')
    text=text.replace(marker,'''  def localFpOpcode():UInt = {
    val decoder=Module(new F2FpOpcodeLocalizer)
    decoder.io.op:=io.in.bits.op
    Cat(0.U(1.W),decoder.io.value)
  }
'''+marker)
    old='(if(predecodedFloat)1.U else io.in.bits.op)'
    if text.count(old)!=2:raise ValueError('unexpected FP opcode interfaces')
    return text.replace(old,'(if(predecodedFloat)1.U else localFpOpcode())')
