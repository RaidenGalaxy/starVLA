package simd

import chisel3._
import chisel3.util._

class DmaCommand extends Bundle {
  val source=UInt(64.W)
  val bank=UInt(2.W)
  val destination=UInt(12.W)
  val beats=UInt(13.W)
  val tag=UInt(32.W)
}
class MemoryReadRequest extends Bundle {
  val address=UInt(64.W)
  val id=UInt(5.W)
}
class MemoryReadResponse extends Bundle {
  val id=UInt(5.W)
  val beat=UInt(2.W)
  val data=UInt(128.W)
  val error=Bool()
}
class DmaCompletion extends Bundle {
  val tag=UInt(32.W)
  val error=Bool()
}

/** 128bit read DMA with 32 credits and 64B bursts. Responses may reorder
  * across IDs and within one burst. A credit returns after all four beats,
  * including padding at the end of the command, have actually been accepted.
  * Source addresses are burst aligned; destination addresses count 128bit
  * local storage words. The separate direct LoadBeat port remains useful for
  * byte-enable and fault-injection tests. */
class ReadDma extends Module {
  val io=IO(new Bundle {
    val command=Flipped(Decoupled(new DmaCommand))
    val request=Decoupled(new MemoryReadRequest)
    val response=Flipped(Decoupled(new MemoryReadResponse))
    val load=Decoupled(new LoadBeat)
    val done=Decoupled(new DmaCompletion)
    val busy=Output(Bool())
    val outstanding=Output(UInt(6.W))
    val requests=Output(UInt(64.W))
    val deliveredBytes=Output(UInt(64.W))
  })
  val active=RegInit(false.B)
  val completed=RegInit(false.B)
  val failed=RegInit(false.B)
  val cmd=Reg(new DmaCommand)
  val issued=RegInit(0.U(13.W))
  val valid=RegInit(VecInit(Seq.fill(32)(false.B)))
  val offsets=Reg(Vec(32,UInt(13.W)))
  val lengths=Reg(Vec(32,UInt(3.W)))
  val received=RegInit(VecInit(Seq.fill(32)(0.U(4.W))))
  val reqCount=RegInit(0.U(64.W))
  val delivered=RegInit(0.U(64.W))
  io.outstanding:=PopCount(valid)
  io.requests:=reqCount;io.deliveredBytes:=delivered
  io.busy:=active||completed
  io.command.ready:= !active && !completed
  when(io.command.fire) {
    assert(io.command.bits.beats=/=0.U)
    assert(io.command.bits.source(5,0)===0.U,"DMA source must be 64-byte aligned")
    assert(io.command.bits.bank<3.U)
    val capacity=Mux(io.command.bits.bank===1.U,2048.U,1024.U)
    assert((io.command.bits.destination+&io.command.bits.beats)<=capacity,"DMA destination exceeds bank")
    cmd:=io.command.bits;issued:=0.U;active:=true.B;failed:=false.B
  }
  val free= ~valid.asUInt
  val next=PriorityEncoder(free)
  val requests=Module(new Queue(new MemoryReadRequest,1,pipe=true))
  requests.io.enq.valid:=active && issued<cmd.beats && free.orR
  requests.io.enq.bits.address:=cmd.source+(issued<<4)
  requests.io.enq.bits.id:=next
  io.request<>requests.io.deq
  when(requests.io.enq.fire) {
    assert(!valid(next))
    valid(next):=true.B;offsets(next):=issued
    lengths(next):=Mux(cmd.beats-issued>=4.U,4.U,cmd.beats-issued)
    received(next):=0.U;issued:=issued+4.U
  }
  when(io.request.fire){reqCount:=reqCount+1.U}
  val id=io.response.bits.id
  val mask=(1.U(4.W)<<io.response.bits.beat)(3,0)
  val known=valid(id)
  val duplicate=(received(id)&mask).orR
  val useful=io.response.bits.beat<lengths(id)
  io.load.valid:=io.response.valid&&known && !duplicate && useful && !io.response.bits.error
  io.load.bits.bank:=cmd.bank
  io.load.bits.address:=(cmd.destination+offsets(id)+io.response.bits.beat)(11,0)
  io.load.bits.strobe:="hffff".U
  io.load.bits.data:=io.response.bits.data
  io.response.ready:=known && !duplicate && (!useful||io.response.bits.error||io.load.ready)
  when(io.response.valid){
    assert(known,"DMA response has no outstanding request")
    assert(!duplicate,"DMA response beat was delivered twice")
  }
  when(io.response.fire) {
    received(id):=received(id)|mask
    when((received(id)|mask).andR){valid(id):=false.B}
    when(io.response.bits.error){failed:=true.B}
  }
  when(io.load.fire){delivered:=delivered+16.U}
  when(active && issued>=cmd.beats && !valid.asUInt.orR) {
    active:=false.B;completed:=true.B
  }
  io.done.valid:=completed;io.done.bits.tag:=cmd.tag;io.done.bits.error:=failed
  when(io.done.fire){completed:=false.B}
}
