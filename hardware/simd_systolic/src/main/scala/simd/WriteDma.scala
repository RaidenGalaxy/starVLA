package simd

import chisel3._
import chisel3.util._

class OutputCommand extends Bundle {
  val address=UInt(64.W);val bytes=UInt(64.W);val tag=UInt(32.W)
}
class OutputPush extends Bundle {val data=UInt(512.W);val bytes=UInt(7.W)}
class OutputPushCommand extends Bundle {
  val bytes=UInt(7.W);val resultTag=UInt(32.W)
  val overrideData=Bool();val data=UInt(512.W)
}
class MemoryWriteData extends Bundle {
  val id=UInt(5.W);val beat=UInt(2.W);val data=UInt(128.W);val enable=UInt(16.W)
}
class MemoryWriteResponse extends Bundle {val id=UInt(5.W);val error=Bool()}

/** Packs actual retired wide results into 64B bursts. The 128bit data
  * channel transfers four beats, with byte enables on a final partial burst.
  * Completion denotes a posted memory acknowledgement; DRAM data completion
  * remains a separate observation supplied by the memory timing model. */
class WriteDma extends Module {
  val io=IO(new Bundle {
    val command=Flipped(Decoupled(new OutputCommand))
    val push=Flipped(Decoupled(new OutputPush))
    val request=Decoupled(new MemoryReadRequest)
    val data=Decoupled(new MemoryWriteData)
    val response=Flipped(Decoupled(new MemoryWriteResponse))
    val done=Decoupled(new DmaCompletion)
    val busy=Output(Bool());val transactionBusy=Output(Bool())
    val requests=Output(UInt(64.W));val payloadBytes=Output(UInt(64.W))
  })
  val collect::header::transfer::ack::Nil=Enum(4)
  val state=RegInit(collect)
  val active=RegInit(false.B);val completed=RegInit(false.B);val failed=RegInit(false.B)
  val cmd=Reg(new OutputCommand)
  val received=RegInit(0.U(64.W));val address=Reg(UInt(64.W))
  val buffer=RegInit(0.U(1024.W));val count=RegInit(0.U(8.W))
  val byteValid=RegInit(0.U(128.W))
  val burst=Reg(UInt(512.W));val enables=Reg(UInt(64.W));val beat=RegInit(0.U(2.W))
  val requests=RegInit(0.U(64.W));val payload=RegInit(0.U(64.W))
  io.requests:=requests;io.payloadBytes:=payload
  io.busy:=active||completed;io.transactionBusy:=state=/=collect
  io.command.ready:= !active && !completed
  when(io.command.fire) {
    assert(io.command.bits.bytes=/=0.U)
    cmd:=io.command.bits;address:=Cat(io.command.bits.address(63,6),0.U(6.W))
    active:=true.B;failed:=false.B;received:=0.U;buffer:=0.U
    count:=io.command.bits.address(5,0);byteValid:=0.U
  }
  val emit=active && state===collect && (count>=64.U || (received===cmd.bytes && count=/=0.U))
  io.push.ready:=active && state===collect && !emit && count<=64.U && received<cmd.bytes
  when(io.push.fire) {
    assert(io.push.bits.bytes=/=0.U && io.push.bits.bytes<=64.U)
    assert(received+&io.push.bits.bytes<=cmd.bytes,"output exceeds declared byte count")
    val mask=((1.U(513.W)<<(io.push.bits.bytes<<3))-1.U)(511,0)
    buffer:=buffer | ((io.push.bits.data&mask)<<(count<<3))
    val validMask=((1.U(65.W)<<io.push.bits.bytes)-1.U)(63,0)
    byteValid:=byteValid | (validMask<<count)
    count:=count+io.push.bits.bytes;received:=received+io.push.bits.bytes
  }
  when(emit) {
    val bytes=Mux(count>=64.U,64.U,count)
    burst:=buffer(511,0);enables:=byteValid(63,0)
    buffer:=buffer>>512;byteValid:=byteValid>>64;count:=count-bytes;beat:=0.U;state:=header
  }
  io.request.valid:=state===header
  io.request.bits.address:=address;io.request.bits.id:=0.U
  when(io.request.fire){requests:=requests+1.U;state:=transfer}
  io.data.valid:=state===transfer
  io.data.bits.id:=0.U;io.data.bits.beat:=beat
  io.data.bits.data:=(burst>>(beat<<7))(127,0)
  io.data.bits.enable:=(enables>>(beat<<4))(15,0)
  when(io.data.fire) {
    payload:=payload+PopCount(io.data.bits.enable)
    beat:=beat+1.U
    when(beat===3.U){state:=ack}
  }
  io.response.ready:=state===ack
  when(io.response.fire) {
    assert(io.response.bits.id===0.U)
    when(io.response.bits.error){failed:=true.B}
    address:=address+64.U;state:=collect
  }
  when(active && received===cmd.bytes && count===0.U && state===collect) {
    active:=false.B;completed:=true.B
  }
  io.done.valid:=completed;io.done.bits.tag:=cmd.tag;io.done.bits.error:=failed
  when(io.done.fire){completed:=false.B}
}
