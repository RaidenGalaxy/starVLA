package simd

import chisel3._

/** A physical single-port bank, composed of 1024x64 ASAP7 macro instances. */
class SramBank(width:Int,depth:Int) extends BlackBox(Map("WIDTH"->width,"DEPTH"->depth)) {
  require(Set(64,128).contains(width) && depth%1024==0)
  override def desiredName="SimdSramBank"
  val io=IO(new Bundle {
    val clock=Input(Clock());val reset=Input(Bool())
    val readEnable=Input(Bool());val readAddress=Input(UInt(13.W))
    val readReady=Output(Bool());val readValid=Output(Bool());val readData=Output(UInt(width.W))
    val writeValid=Input(Bool());val writeReady=Output(Bool())
    val writeAddress=Input(UInt(13.W));val writeData=Input(UInt(width.W));val writeStrobe=Input(UInt((width/8).W))
    val busy=Output(Bool());val writeDone=Output(Bool());val rmwReadBytes=Output(UInt(8.W));val writeBytes=Output(UInt(8.W))
  })
}
