package simd

import chisel3._
import circt.stage.ChiselStage

/** Combinational probe for exhaustive raw-code source-expansion checks. */
class FloatExpansionProbe extends RawModule {
  val io=IO(new Bundle {
    val code=Input(UInt(16.W));val fmt=Input(UInt(4.W));val value=Output(UInt(32.W))
  })
  io.value:=ExpandFloat(io.code.pad(32),io.fmt)
}
object GenerateExpansionProbe extends App {
  ChiselStage.emitSystemVerilogFile(new FloatExpansionProbe,
    Array("--target-dir",args(0)),Array("--disable-all-randomization","-strip-debug-info"))
}
