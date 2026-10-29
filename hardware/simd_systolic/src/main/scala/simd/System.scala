package simd

import chisel3._
import chisel3.util._

class CdcFifo(width:Int,addr:Int) extends BlackBox(Map("WIDTH"->width,"ADDR"->addr)) {
  override def desiredName="AsyncFifo"
  val io=IO(new Bundle {
    val wclk=Input(Clock());val wrst=Input(Bool());val rclk=Input(Clock());val rrst=Input(Bool())
    val wvalid=Input(Bool());val wready=Output(Bool());val wdata=Input(UInt(width.W))
    val rvalid=Output(Bool());val rready=Input(Bool());val rdata=Output(UInt(width.W))
  })
}
class ResetSynchronizer extends BlackBox {
  override def desiredName="SimdResetSynchronizer"
  val io=IO(new Bundle {
    val clock=Input(Clock());val externalReset=Input(Bool());val value=Output(Bool())
  })
}
class ToggleSynchronizer extends BlackBox {
  override def desiredName="SimdToggleSynchronizer"
  val io=IO(new Bundle {
    val clock=Input(Clock());val reset=Input(Bool());val data=Input(Bool());val value=Output(Bool())
  })
}
class LoadBeat extends Bundle {
  val bank=UInt(2.W);val address=UInt(12.W);val data=UInt(128.W);val strobe=UInt(16.W)
}
class TileCommand(cfg:Config) extends Bundle {
  val fmt=UInt(4.W);val contexts=UInt(5.W);val k=UInt(4.W);val clear=Bool()
  val rowMask=UInt(cfg.rows.W);val masks=Vec(cfg.cols,UInt(16.W));val tag=UInt(32.W)
}
class WavePacket(cfg:Config) extends Bundle {
  val a=Vec(cfg.rows,UInt(64.W));val b=Vec(cfg.cols,UInt(64.W))
  val fmt=UInt(4.W);val ctx=UInt(4.W);val clear=Bool();val rowMask=UInt(cfg.rows.W)
  val masks=Vec(cfg.cols,UInt(16.W));val tag=UInt(32.W);val last=Bool()
}
class ReadCommand extends Bundle {val ctx=UInt(4.W);val row=UInt(4.W);val col=UInt(4.W);val tag=UInt(32.W)}
class TileDone extends Bundle {
  val cycles=UInt(64.W);val macs=UInt(64.W);val overflow=UInt(64.W);val exceptions=UInt(64.W);val tag=UInt(32.W)
  val vectors=UInt(64.W);val feedbackStalls=UInt(64.W);val inputStalls=UInt(64.W)
}
class AuxiliaryCommand extends Bundle {
  val request=new Request
  val aggregate=Bool();val clearAggregate=Bool();val aggregateIndex=UInt(13.W)
  val useScale=Bool();val scaleAddress=UInt(10.W)
}

/** Tile command hardware, 64 KiB explicit storage, independently clocked
  * core and memory. The host GEMM driver tiles arbitrary M/N/K into commands.
  * SRAM layout is [k][context][row/column], each entry one packed 64-bit word. */
class MemoryPeek extends Bundle {val bank=UInt(2.W);val address=UInt(12.W)}
class AggregateInspection extends Bundle {
  val address=UInt(13.W);val write=Bool();val data=UInt(64.W);val strobe=UInt(8.W)
}
class TileSystem(cfg:Config) extends RawModule {
  require(cfg.rows<=8 && cfg.cols<=8 && isPow2(cfg.rows) && isPow2(cfg.cols))
  val io=IO(new Bundle {
    val coreClock=Input(Clock());val memoryClock=Input(Clock())
    val coreReset=Input(AsyncReset());val memoryReset=Input(AsyncReset())
    val load=Flipped(Decoupled(new LoadBeat));val command=Flipped(Decoupled(new TileCommand(cfg)))
    val seed=Flipped(Decoupled(new ArraySeed))
    val read=Flipped(Decoupled(new ReadCommand));val result=Decoupled(new Response)
    val done=Decoupled(new TileDone);val busy=Output(Bool());val coreActive=Output(Bool())
    val inputTileCaptured=Output(Bool())
    val peek=Flipped(Decoupled(new MemoryPeek));val peekResult=Decoupled(UInt(128.W));val memoryIdle=Output(Bool())
    val sramReadBytes=Output(UInt(64.W));val sramWriteBytes=Output(UInt(64.W))
    val aggregateReadBytes=Output(UInt(64.W));val aggregateWriteBytes=Output(UInt(64.W))
    val aggregateInspection=Flipped(Decoupled(new AggregateInspection));val aggregateInspectionResult=Decoupled(UInt(64.W))
    val auxiliary=Flipped(Decoupled(new AuxiliaryCommand))
    val dmaCommand=Flipped(Decoupled(new DmaCommand));val dmaDone=Decoupled(new DmaCompletion)
    val memoryRequest=Decoupled(new MemoryReadRequest)
    val memoryResponse=Flipped(Decoupled(new MemoryReadResponse))
    val dmaOutstanding=Output(UInt(6.W));val dmaReadRequests=Output(UInt(64.W))
    val dmaPayloadBytes=Output(UInt(64.W))
    val outputStart=Flipped(Decoupled(new OutputCommand))
    val outputPush=Flipped(Decoupled(new OutputPushCommand));val outputDone=Decoupled(new DmaCompletion)
    val writeRequest=Decoupled(new MemoryReadRequest);val writeData=Decoupled(new MemoryWriteData)
    val writeResponse=Flipped(Decoupled(new MemoryWriteResponse))
    val dmaWriteRequests=Output(UInt(64.W));val dmaWritePayloadBytes=Output(UInt(64.W))
  })
  // Assert immediately; release only after two edges in each receiving domain.
  def releasedReset(clock:Clock,external:AsyncReset):AsyncReset = {
    val sync=Module(new ResetSynchronizer)
    sync.io.clock:=clock;sync.io.externalReset:=external.asBool;sync.io.value.asAsyncReset
  }
  // A reset at either external endpoint aborts the whole transaction epoch.
  // Both FIFO endpoints assert together and release in their own clocks.
  val epochReset=(io.coreReset.asBool || io.memoryReset.asBool).asAsyncReset
  val coreReset=releasedReset(io.coreClock,epochReset)
  val memoryReset=releasedReset(io.memoryClock,epochReset)
  val packetWidth=(new WavePacket(cfg)).getWidth
  def fifo(w:Int,a:Int,reverse:Boolean=false):CdcFifo = {
    val q=Module(new CdcFifo(w,a))
    q.io.wclk:=(if(reverse)io.coreClock else io.memoryClock)
    q.io.rclk:=(if(reverse)io.memoryClock else io.coreClock)
    q.io.wrst:=(if(reverse)coreReset.asBool else memoryReset.asBool)
    q.io.rrst:=(if(reverse)memoryReset.asBool else coreReset.asBool);q
  }
  val waves=fifo(packetWidth,8);val reads=fifo((new ReadCommand).getWidth,4)
  val results=fifo((new Response).getWidth,4,true);val dones=fifo((new TileDone).getWidth,2,true)
  val auxRequests=fifo((new Request).getWidth,4)
  val auxResults=fifo((new Response).getWidth,4,true)
  val seeds=fifo((new ArraySeed).getWidth,2);val seedAcks=fifo(1,2,true)
  // Complete the memory-domain handshake only after the context write has
  // committed in the core. A following command cannot overtake its seed.
  withClockAndReset(io.memoryClock,memoryReset) {
    val pending=RegInit(false.B)
    seeds.io.wvalid:=io.seed.valid && !pending;seeds.io.wdata:=io.seed.bits.asUInt
    when(seeds.io.wvalid&&seeds.io.wready){pending:=true.B}
    io.seed.ready:=pending && seedAcks.io.rvalid
    seedAcks.io.rready:=pending && io.seed.valid
    when(io.seed.fire){pending:=false.B}
  }
  reads.io.wvalid:=io.read.valid;reads.io.wdata:=io.read.bits.asUInt;io.read.ready:=reads.io.wready
  dones.io.rready:=io.done.ready;io.done.valid:=dones.io.rvalid;io.done.bits:=dones.io.rdata.asTypeOf(new TileDone)
  val runToggle=Wire(Bool())
  withClockAndReset(io.memoryClock,memoryReset) {
    val writer=Module(new WriteDma)
    io.outputStart<>writer.io.command;io.outputDone<>writer.io.done
    io.writeData<>writer.io.data;writer.io.response<>io.writeResponse
    io.dmaWriteRequests:=writer.io.requests;io.dmaWritePayloadBytes:=writer.io.payloadBytes
    val retiredData=Reg(UInt(512.W));val retiredTag=Reg(UInt(32.W));val retiredValid=RegInit(false.B)
    val outputPending=RegInit(false.B);val outputData=Reg(UInt(512.W));val outputBytes=Reg(UInt(7.W))
    writer.io.push.valid:=outputPending;writer.io.push.bits.data:=outputData;writer.io.push.bits.bytes:=outputBytes
    io.outputPush.ready:= !outputPending && retiredValid
    when(writer.io.push.fire){outputPending:=false.B}
    when(io.outputPush.fire) {
      assert(io.outputPush.bits.resultTag===retiredTag,"output DMA result tag mismatch")
      outputData:=Mux(io.outputPush.bits.overrideData,io.outputPush.bits.data,retiredData)
      outputBytes:=io.outputPush.bits.bytes;outputPending:=true.B
    }
    when(io.result.fire){retiredData:=io.result.bits.data;retiredTag:=io.result.bits.tag;retiredValid:=true.B}
    val dma=Module(new ReadDma)
    io.memoryRequest<>dma.io.request;dma.io.response<>io.memoryResponse
    // One write transaction reserves the memory front end after all accepted
    // reads have drained. A subsequent read descriptor waits for that posted
    // write acknowledgement; at most 32 external transactions are outstanding.
    io.writeRequest.valid:=writer.io.request.valid && !dma.io.busy
    io.writeRequest.bits:=writer.io.request.bits
    writer.io.request.ready:=io.writeRequest.ready && !dma.io.busy
    io.dmaOutstanding:=dma.io.outstanding
    io.dmaReadRequests:=dma.io.requests;io.dmaPayloadBytes:=dma.io.deliveredBytes
    val loadArbiter=Module(new RRArbiter(new LoadBeat,2))
    loadArbiter.io.in(0)<>io.load;loadArbiter.io.in(1)<>dma.io.load
    val load=loadArbiter.io.out
    val a=Module(new SramBank(128,1024));val b=Module(new SramBank(128,2048))
    val c=Module(new SramBank(128,1024));val aggregates=Module(new SramBank(64,8192))
    val banks=Seq(a,b,c)
    for(bank<-banks:+aggregates){bank.io.clock:=io.memoryClock;bank.io.reset:=memoryReset.asBool}
    val storageIdle= !(banks.map(_.io.busy).reduce(_||_) || aggregates.io.busy)
    io.memoryIdle:=storageIdle
    io.dmaDone.valid:=dma.io.done.valid && storageIdle;io.dmaDone.bits:=dma.io.done.bits
    dma.io.done.ready:=io.dmaDone.ready && storageIdle
    val idle::fetch::push::waitDone::Nil=Enum(4)
    val state=RegInit(idle);val cmd=Reg(new TileCommand(cfg));val toggle=RegInit(false.B)
    runToggle:=toggle
    io.busy:=state=/=idle
    io.inputTileCaptured:=state===waitDone
    val auxIdle::auxPrepare::auxSend::auxWait::auxCommit::Nil=Enum(5)
    val auxState=RegInit(auxIdle);val auxCommand=Reg(new AuxiliaryCommand)
    val scalePending=RegInit(false.B);val aggregatePending=RegInit(false.B)
    val initialAggregate=Reg(UInt(64.W));val aggregateCommitted=RegInit(false.B)
    val inspectPending=RegInit(false.B);val inspectWrite=Reg(Bool())
    val inspectResult=Reg(UInt(64.W));val inspectValid=RegInit(false.B)
    val peekPending=RegInit(false.B);val peekBank=Reg(UInt(2.W))
    val peekData=Reg(UInt(128.W));val peekValid=RegInit(false.B)
    val debugBusy=inspectPending || inspectValid || peekPending || peekValid
    val inspection=io.aggregateInspection
    inspection.ready:=state===idle && auxState===auxIdle && !debugBusy && storageIdle &&
      !load.valid && !dma.io.busy && !io.command.valid && !io.auxiliary.valid
    io.aggregateInspectionResult.valid:=inspectValid;io.aggregateInspectionResult.bits:=inspectResult
    when(io.aggregateInspectionResult.fire){inspectValid:=false.B}
    when(inspection.fire){inspectPending:=true.B;inspectWrite:=inspection.bits.write}
    val aggregateAuxRead=io.auxiliary.fire && io.auxiliary.bits.aggregate && !io.auxiliary.bits.clearAggregate
    aggregates.io.readEnable:=aggregateAuxRead || (inspection.fire && !inspection.bits.write)
    aggregates.io.readAddress:=Mux(inspection.fire,inspection.bits.address,io.auxiliary.bits.aggregateIndex)
    val auxiliaryWrite=auxState===auxWait && auxCommand.aggregate && auxResults.io.rvalid
    aggregates.io.writeValid:=auxiliaryWrite || (inspection.valid && inspection.ready && inspection.bits.write)
    aggregates.io.writeAddress:=Mux(auxiliaryWrite,auxCommand.aggregateIndex,inspection.bits.address)
    aggregates.io.writeData:=Mux(auxiliaryWrite,auxResults.io.rdata.asTypeOf(new Response).data(63,0),inspection.bits.data)
    aggregates.io.writeStrobe:=Mux(auxiliaryWrite,255.U,inspection.bits.strobe)
    when(inspectPending && Mux(inspectWrite,aggregates.io.writeDone,aggregates.io.readValid)) {
      inspectResult:=Mux(inspectWrite,0.U,aggregates.io.readData);inspectValid:=true.B;inspectPending:=false.B
    }
    val sramReadBytes=RegInit(0.U(64.W));val sramWriteBytes=RegInit(0.U(64.W))
    val aggregateReadBytes=RegInit(0.U(64.W));val aggregateWriteBytes=RegInit(0.U(64.W))
    io.sramReadBytes:=sramReadBytes;io.sramWriteBytes:=sramWriteBytes
    io.aggregateReadBytes:=aggregateReadBytes;io.aggregateWriteBytes:=aggregateWriteBytes
    aggregateReadBytes:=aggregateReadBytes+Mux(aggregates.io.readEnable && aggregates.io.readReady,8.U,0.U)+aggregates.io.rmwReadBytes
    aggregateWriteBytes:=aggregateWriteBytes+aggregates.io.writeBytes
    sramWriteBytes:=sramWriteBytes+a.io.writeBytes+b.io.writeBytes+c.io.writeBytes
    io.command.ready:=state===idle && auxState===auxIdle && !load.valid && !dma.io.busy && storageIdle && !debugBusy && !inspection.valid && !io.peek.valid
    io.auxiliary.ready:=auxState===auxIdle && state===idle && !load.valid && !io.command.valid && !dma.io.busy && storageIdle && !debugBusy
    when(io.auxiliary.fire){
      auxCommand:=io.auxiliary.bits;auxState:=auxPrepare
      initialAggregate:=0.U;scalePending:=io.auxiliary.bits.useScale
      aggregatePending:=io.auxiliary.bits.aggregate && !io.auxiliary.bits.clearAggregate
      aggregateCommitted:=false.B
    }
    when(auxState===auxPrepare && aggregatePending && aggregates.io.readValid){
      initialAggregate:=aggregates.io.readData;aggregatePending:=false.B
    }
    when(auxState===auxPrepare && !scalePending && !aggregatePending){auxState:=auxSend}
    val dmaAdmit=(state===idle || state===waitDone) && auxState===auxIdle && !writer.io.transactionBusy && !debugBusy && storageIdle
    dma.io.command.valid:=io.dmaCommand.valid && dmaAdmit
    dma.io.command.bits:=io.dmaCommand.bits
    io.dmaCommand.ready:=dma.io.command.ready && dmaAdmit
    val loadAllowed=(state===idle || state===waitDone) && auxState===auxIdle && !debugBusy
    load.ready:=loadAllowed && MuxLookup(load.bits.bank,false.B)(banks.zipWithIndex.map{case(bank,i)=>i.U->bank.io.writeReady})
    for((bank,i)<-banks.zipWithIndex){
      bank.io.writeValid:=load.valid && loadAllowed && load.bits.bank===i.U
      bank.io.writeAddress:=load.bits.address;bank.io.writeData:=load.bits.data;bank.io.writeStrobe:=load.bits.strobe
    }
    when(load.fire) {
      assert(load.bits.bank<3.U)
      assert(load.bits.address<Mux(load.bits.bank===1.U,2048.U,1024.U))
    }
    val ctx=RegInit(0.U(4.W));val k=RegInit(0.U(4.W));val word=RegInit(0.U(3.W))
    val av=Reg(Vec(cfg.rows,UInt(64.W)));val bv=Reg(Vec(cfg.cols,UInt(64.W)))
    val nwords=math.max((cfg.rows+1)/2,(cfg.cols+1)/2)
    val issue=state===fetch && word<nwords.U && a.io.readReady && b.io.readReady
    val slot=k*cmd.contexts+ctx
    val peekReady=MuxLookup(io.peek.bits.bank,false.B)(banks.zipWithIndex.map{case(bank,i)=>i.U->bank.io.readReady})
    io.peek.ready:=state===idle && auxState===auxIdle && !debugBusy && storageIdle && !load.valid &&
      !dma.io.busy && !io.command.valid && !io.auxiliary.valid && !inspection.valid && peekReady
    io.peekResult.valid:=peekValid;io.peekResult.bits:=peekData
    when(io.peekResult.fire){peekValid:=false.B}
    when(io.peek.fire){peekPending:=true.B;peekBank:=io.peek.bits.bank}
    val aRead=(issue && word<((cfg.rows+1)/2).U) || (io.peek.fire && io.peek.bits.bank===0.U)
    val bRead=(issue && word<((cfg.cols+1)/2).U) || (io.peek.fire && io.peek.bits.bank===1.U)
    val cRead=(io.auxiliary.fire && io.auxiliary.bits.useScale) || (io.peek.fire && io.peek.bits.bank===2.U)
    a.io.readEnable:=aRead;b.io.readEnable:=bRead;c.io.readEnable:=cRead
    a.io.readAddress:=Mux(issue,slot*((cfg.rows+1)/2).U+word,io.peek.bits.address)
    b.io.readAddress:=Mux(issue,slot*((cfg.cols+1)/2).U+word,io.peek.bits.address)
    c.io.readAddress:=Mux(io.auxiliary.fire,io.auxiliary.bits.scaleAddress,io.peek.bits.address)
    val ar=a.io.readData;val br=b.io.readData;val cr=c.io.readData
    sramReadBytes:=sramReadBytes+(PopCount(Seq(aRead,bRead,cRead))<<4)+a.io.rmwReadBytes+b.io.rmwReadBytes+c.io.rmwReadBytes
    val peekReturned=MuxLookup(peekBank,false.B)(banks.zipWithIndex.map{case(bank,i)=>i.U->bank.io.readValid})
    when(peekPending && peekReturned){
      peekData:=MuxLookup(peekBank,0.U)(Seq(0.U->ar,1.U->br,2.U->cr));peekPending:=false.B;peekValid:=true.B
    }
    when(auxState===auxPrepare && scalePending && c.io.readValid) {
      auxCommand.request.a:=Cat(cr(95,64),cr(31,0))
      auxCommand.request.b:=Cat(cr(127,96),cr(63,32));scalePending:=false.B
    }
    val auxiliaryRequest=WireDefault(auxCommand.request)
    when(auxCommand.aggregate){auxiliaryRequest.acc:=initialAggregate.pad(512)}
    auxRequests.io.wvalid:=auxState===auxSend;auxRequests.io.wdata:=auxiliaryRequest.asUInt
    when(auxRequests.io.wvalid&&auxRequests.io.wready){auxState:=auxWait}
    when(auxiliaryWrite && aggregates.io.writeReady){auxState:=auxCommit}
    when(auxState===auxCommit && aggregates.io.writeDone){aggregateCommitted:=true.B}
    val arb=Module(new RRArbiter(new Response,2))
    arb.io.in(0).valid:=results.io.rvalid;arb.io.in(0).bits:=results.io.rdata.asTypeOf(new Response)
    results.io.rready:=arb.io.in(0).ready
    val auxiliaryReturnAllowed= !auxCommand.aggregate || aggregateCommitted
    arb.io.in(1).valid:=auxResults.io.rvalid && auxiliaryReturnAllowed
    arb.io.in(1).bits:=auxResults.io.rdata.asTypeOf(new Response)
    auxResults.io.rready:=arb.io.in(1).ready && auxiliaryReturnAllowed
    io.result<>arb.io.out
    when(arb.io.in(1).fire) {
      assert(auxState===auxWait || auxState===auxCommit)
      auxState:=auxIdle;aggregateCommitted:=false.B
    }
    val ret=ShiftRegister(issue,cfg.sramReadLatency,false.B,true.B)
    val wi=if(cfg.sramReadLatency==1)RegEnable(word,issue)else Pipes.delay(word,cfg.sramReadLatency,true.B)
    when(io.command.fire) {
      assert(io.command.bits.contexts>=1.U && io.command.bits.contexts<=16.U)
      assert(io.command.bits.k>=1.U && io.command.bits.k<=8.U)
      assert(io.command.bits.fmt<=10.U)
      cmd:=io.command.bits;ctx:=0.U;k:=0.U;word:=0.U;state:=fetch
    }
    when(issue){word:=word+1.U}
    when(ret) {
      for(i<-0 until cfg.rows) when(wi===(i/2).U){av(i):=ar(64*(i%2)+63,64*(i%2))}
      for(i<-0 until cfg.cols) when(wi===(i/2).U){bv(i):=br(64*(i%2)+63,64*(i%2))}
      when(wi===(nwords-1).U){state:=push}
    }
    val packet=Wire(new WavePacket(cfg));packet.a:=av;packet.b:=bv
    packet.fmt:=cmd.fmt;packet.ctx:=ctx;packet.clear:=cmd.clear && k===0.U
    packet.rowMask:=cmd.rowMask;packet.masks:=cmd.masks
    packet.tag:=cmd.tag+k*cmd.contexts+ctx
    packet.last:=ctx===cmd.contexts-1.U && k===cmd.k-1.U
    waves.io.wdata:=packet.asUInt;waves.io.wvalid:=state===push
    when(waves.io.wvalid&&waves.io.wready) {
      word:=0.U
      when(packet.last){state:=waitDone;toggle:= ~toggle}
      .elsewhen(ctx===cmd.contexts-1.U){ctx:=0.U;k:=k+1.U;state:=fetch}
      .otherwise{ctx:=ctx+1.U;state:=fetch}
    }
    when(io.done.fire){state:=idle}
  }
  withClockAndReset(io.coreClock,coreReset) {
    // Identical synchronous-reset arithmetic/state cells in standalone PPA
    // and in this system. The reset source has already synchronized release
    // to this clock; CDC FIFOs retain asynchronous assertion.
    val array=withReset(coreReset.asBool){Module(new SystolicArray(cfg))}
    val auxiliary=withReset(coreReset.asBool){Module(new SimdAlu(cfg,predecodedFloat=cfg.stateReadStages!=0))}
    if(cfg.stateReadStages==0) {
      auxiliary.io.in.valid:=auxRequests.io.rvalid
      auxiliary.io.in.bits:=auxRequests.io.rdata.asTypeOf(new Request)
      auxRequests.io.rready:=auxiliary.io.in.ready
    }else {
      val capacity=cfg.stages+2
      val credits=RegInit(capacity.U(log2Ceil(capacity+1).W))
      auxRequests.io.rready:=credits=/=0.U || auxiliary.io.out.fire
      val accepted=auxRequests.io.rvalid && auxRequests.io.rready
      val raw=auxRequests.io.rdata.asTypeOf(new Request)
      val request=RegNext(raw);val sources=RegNext(ExpandSources(raw))
      val preparation=Module(new PreparationControlReplicas)
      preparation.io.clock:=io.coreClock;preparation.io.fmt:=raw.fmt;preparation.io.op:=raw.op
      val prepared=RegNext(PrepareOperands(request,source=Some(sources),control=Some(preparation.io.value)))
      auxiliary.io.in.valid:=RegNext(RegNext(accepted,false.B),false.B);auxiliary.io.in.bits:=prepared.request
      auxiliary.io.floating.get:=prepared.floating
      when(accepted && !auxiliary.io.out.fire){credits:=credits-1.U}
      when(!accepted && auxiliary.io.out.fire){credits:=credits+1.U}
      assert(credits<=capacity.U)
      when(auxiliary.io.in.valid){assert(auxiliary.io.in.ready,"reserved auxiliary operand could not advance")}
    }
    auxiliary.io.out.ready:=auxResults.io.wready
    auxResults.io.wvalid:=auxiliary.io.out.valid;auxResults.io.wdata:=auxiliary.io.out.bits.asUInt
    val commandSync=Module(new ToggleSynchronizer)
    commandSync.io.clock:=io.coreClock;commandSync.io.reset:=coreReset.asBool;commandSync.io.data:=runToggle
    val sync2=commandSync.io.value
    val seen=RegInit(false.B);val running=RegInit(false.B);val drain=RegInit(false.B)
    val timer=RegInit(0.U(64.W));val beginMacs=Reg(UInt(64.W));val tag=Reg(UInt(32.W))
    val beginOverflow=Reg(UInt(64.W));val beginException=Reg(UInt(64.W))
    val beginVectors=Reg(UInt(64.W))
    val feedbackStalls=RegInit(0.U(64.W));val inputStalls=RegInit(0.U(64.W))
    val readOutstanding=WireDefault(false.B)
    val wave=waves.io.rdata.asTypeOf(new WavePacket(cfg))
    val start=sync2=/=seen && !readOutstanding
    io.coreActive:=running || drain || array.io.busy
    val arrayIdle=RegNext(!array.io.busy,false.B)
    val seedInFlight=if(cfg.seedWriteLatency==1)false.B else RegNext(array.io.seed.valid,false.B)
    val seedAllowed= !running && !drain && arrayIdle && !start && !readOutstanding && !seedInFlight
    seeds.io.rready:=seedAllowed && seedAcks.io.wready
    array.io.seed.valid:=seeds.io.rvalid && seeds.io.rready
    array.io.seed.bits:=seeds.io.rdata.asTypeOf(new ArraySeed)
    seedAcks.io.wvalid:=(if(cfg.seedWriteLatency==1)array.io.seed.valid else RegNext(array.io.seed.valid,false.B))
    seedAcks.io.wdata:=1.U
    when(seedAcks.io.wvalid){assert(seedAcks.io.wready,"seed acknowledgement lost its reserved FIFO slot")}
    when(start){
      seen:=sync2;running:=true.B;timer:=0.U;beginMacs:=array.io.macs;tag:=wave.tag
      beginOverflow:=array.io.overflowEvents;beginException:=array.io.exceptionOps
      beginVectors:=array.io.completions;feedbackStalls:=0.U;inputStalls:=0.U
    }
    when(running||(drain && array.io.busy)){timer:=(if(cfg.stages==2)timer+1.U else FastCarry.add(timer,1.U,64))}
    // Contexts fewer than the feedback interval require bubbles between K rounds.
    when(array.io.inject && wave.last){running:=false.B;drain:=true.B}
    // A countdown per context permits arbitrary small context groups safely.
    val cooldown=RegInit(VecInit(Seq.fill(16)(0.U(5.W))))
    for(i<-0 until 16) when(cooldown(i)=/=0.U){cooldown(i):=cooldown(i)-1.U}
    val contextReady=cooldown(wave.ctx)===0.U
    when(running && waves.io.rvalid && !contextReady){
      feedbackStalls:=(if(cfg.stages==2)feedbackStalls+1.U else FastCarry.add(feedbackStalls,1.U,64))
    }
    when(running && !waves.io.rvalid){
      inputStalls:=(if(cfg.stages==2)inputStalls+1.U else FastCarry.add(inputStalls,1.U,64))
    }
    waves.io.rready:=running && contextReady
    array.io.inject:=waves.io.rvalid&&running&&contextReady
    when(array.io.inject){cooldown(wave.ctx):=(cfg.feedbackInterval-1).U}
    array.io.a:=wave.a;array.io.b:=wave.b;array.io.masks:=wave.masks
    array.io.rowMask:=wave.rowMask;array.io.fmt:=wave.fmt;array.io.ctx:=wave.ctx
    array.io.tag:=wave.tag;array.io.clear:=wave.clear
    val done=Wire(new TileDone);done.cycles:=timer;done.macs:=(if(cfg.stages==2)array.io.macs-beginMacs else FastCarry.add(array.io.macs,~beginMacs,64,true.B));done.tag:=tag
    done.overflow:=(if(cfg.stages==2)array.io.overflowEvents-beginOverflow else FastCarry.add(array.io.overflowEvents,~beginOverflow,64,true.B));done.exceptions:=(if(cfg.stages==2)array.io.exceptionOps-beginException else FastCarry.add(array.io.exceptionOps,~beginException,64,true.B))
    done.vectors:=(if(cfg.stages==2)array.io.completions-beginVectors else FastCarry.add(array.io.completions,~beginVectors,64,true.B))
    done.feedbackStalls:=feedbackStalls;done.inputStalls:=inputStalls
    dones.io.wdata:=done.asUInt;dones.io.wvalid:=drain && !array.io.busy
    when(dones.io.wvalid&&dones.io.wready){drain:=false.B}
    val rc=reads.io.rdata.asTypeOf(new ReadCommand)
    array.io.readCtx:=rc.ctx;array.io.readRow:=rc.row;array.io.readCol:=rc.col
    val response=WireDefault(0.U.asTypeOf(new Response))
    response.data:=array.io.readData;response.flags:=array.io.readFlags
    if(cfg.arrayReadLatency==0) {
      reads.io.rready:= !array.io.busy && !running && results.io.wready
      val readFire=reads.io.rvalid&&reads.io.rready;array.io.readEnable:=readFire
      when(readFire){assert(rc.row<cfg.rows.U && rc.col<cfg.cols.U)}
      response.ctx:=rc.ctx;response.tag:=rc.tag
      results.io.wvalid:=readFire;results.io.wdata:=response.asUInt
    }else {
      val capacity=cfg.arrayReadLatency+2
      val credits=RegInit(capacity.U(log2Ceil(capacity+1).W))
      readOutstanding:=credits=/=capacity.U
      reads.io.rready:= !array.io.busy && !running && !start && credits=/=0.U
      val readFire=reads.io.rvalid&&reads.io.rready;array.io.readEnable:=readFire
      when(readFire){assert(rc.row<cfg.rows.U && rc.col<cfg.cols.U)}
      val metadata=Pipes.delay(rc,cfg.arrayReadLatency,true.B)
      val valid=(0 until cfg.arrayReadLatency).foldLeft(readFire)((v,_)=>RegNext(v,false.B))
      response.ctx:=metadata.ctx;response.tag:=metadata.tag
      val buffered=Module(new Queue(new Response,capacity,flow=true))
      buffered.io.enq.valid:=valid;buffered.io.enq.bits:=response
      when(valid){assert(buffered.io.enq.ready,"result collector lost a reserved credit")}
      results.io.wvalid:=buffered.io.deq.valid;results.io.wdata:=buffered.io.deq.bits.asUInt
      buffered.io.deq.ready:=results.io.wready
      when(readFire && !buffered.io.deq.fire){credits:=credits-1.U}
      when(!readFire && buffered.io.deq.fire){credits:=credits+1.U}
    }
  }
}
