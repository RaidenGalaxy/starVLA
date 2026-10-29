package simd

import chisel3._
import chisel3.util._

class PeRequest extends Bundle {
  val a=UInt(64.W);val b=UInt(64.W);val fmt=UInt(4.W);val op=UInt(3.W)
  val mask=UInt(16.W);val ctx=UInt(4.W);val tag=UInt(32.W);val clear=Bool()
}
class StateSeed extends Bundle {
  val ctx=UInt(4.W);val data=UInt(512.W);val flags=UInt(5.W)
}
class ArraySeed extends Bundle {
  val row=UInt(4.W);val col=UInt(4.W);val value=new StateSeed
}
class ContextReadReplicas extends BlackBox {
  override def desiredName="SimdContextReadReplicas"
  val io=IO(new Bundle {
    val clock=Input(Clock());val ctx=Input(UInt(4.W));val initialized=Input(UInt(16.W))
    val clear=Input(Bool());val value=Output(UInt(256.W))
  })
}
class ObservationState extends BlackBox {
  override def desiredName="SimdObservationState"
  val io=IO(new Bundle {
    val clock=Input(Clock());val reset=Input(Bool())
    val address=Input(UInt(16.W));val commitAddress=Input(UInt(16.W))
    val commit=Input(Bool());val seedAddress=Input(UInt(16.W));val selected=Output(UInt(16.W))
  })
}
class StateWriteSelect extends BlackBox {
  override def desiredName="SimdStateWriteSelect"
  val io=IO(new Bundle {
    val address=Input(UInt(16.W));val commit=Input(Bool())
    val write=Output(UInt(16.W))
  })
}
class PeSeedWord extends BlackBox {
  override def desiredName="SimdPeSeedWord"
  val io=IO(new Bundle {
    val clock=Input(Clock());val reset=Input(Bool());val ctx=Input(UInt(4.W));val valid=Input(Bool())
    val data=Input(UInt(32.W));val value=Output(UInt(32.W));val selected=Output(UInt(16.W))
  })
}
class PeCommitWord(streaming:Boolean) extends BlackBox {
  override def desiredName=if(streaming)"SimdPeCommitWord"else"SimdPeElasticCommitWord"
  val io=IO(new Bundle {
    val clock=Input(Clock());val reset=Input(Bool());val ready=Input(Bool());val valid=Input(Bool())
    val data=Input(UInt(32.W));val ctx=Input(UInt(4.W));val value=Output(UInt(32.W))
    val address=Output(UInt(16.W));val validOut=Output(Bool())
  })
}
class ElasticPreparationWord extends BlackBox {
  override def desiredName="SimdElasticPreparationWord"
  val io=IO(new Bundle {
    val clock=Input(Clock());val reset=Input(Bool());val accepted=Input(Bool());val blocked=Input(Bool())
    val unavailable=Input(UInt(16.W));val initialized=Input(UInt(16.W))
    val ctx=Input(UInt(4.W));val fmt=Input(UInt(4.W));val op=Input(UInt(3.W));val clear=Input(Bool())
    val data=Input(UInt(32.W));val value=Output(UInt(32.W));val controls=Output(UInt(19.W))
    val selected=Output(UInt(16.W));val capture=Output(Bool());val state=Output(UInt(6.W))
  })
}
class SimdPE(cfg:Config,streaming:Boolean=false) extends Module with RequireSyncReset {
  val io=IO(new Bundle {
    val in=Flipped(Decoupled(new PeRequest));val out=Decoupled(new Response)
    val readEnable=Input(Bool());val readCtx=Input(UInt(4.W));val readData=Output(UInt(512.W))
    val readFlags=Output(UInt(5.W));val pending=Output(UInt(16.W))
    val seed=Flipped(Valid(new StateSeed))
  })
  val accum=Reg(Vec(cfg.contexts,UInt(512.W)))
  val initialized=RegInit(VecInit(Seq.fill(cfg.contexts)(false.B)))
  val flags=RegInit(VecInit(Seq.fill(cfg.contexts)(0.U(5.W))))
  val pending=RegInit(VecInit(Seq.fill(cfg.contexts)(false.B)))
  val seedValid=if(cfg.seedWriteLatency==1)io.seed.valid else RegNext(io.seed.valid,false.B)
  val seedContext=if(cfg.seedWriteLatency==1)io.seed.bits.ctx else RegNext(io.seed.bits.ctx)
  val seedFlags=if(cfg.seedWriteLatency==1)io.seed.bits.flags else RegNext(io.seed.bits.flags)
  val seedInFlight=if(cfg.seedWriteLatency==1)false.B else seedValid
  val seedWords=if(cfg.seedWriteLatency==1)Seq.empty else Seq.tabulate(16) {word=>
    val bank=Module(new PeSeedWord)
    bank.io.clock:=clock;bank.io.reset:=reset.asBool;bank.io.ctx:=io.seed.bits.ctx
    bank.io.valid:=io.seed.valid;bank.io.data:=io.seed.bits.data(32*word+31,32*word)
    bank
  }
  val fixedCommit=cfg.stages>2 && !streaming
  val responsesPending=WireDefault(false.B)
  val alu=Module(new SimdAlu(cfg,streaming,predecodedFloat=cfg.stateReadStages!=0,reservedOutput=fixedCommit))
  val operands=Wire(Decoupled(new Request))
  val stateIssue=WireDefault(io.in.fire)
  val issueContext=WireDefault(io.in.bits.ctx)
  val issueClear=WireDefault(io.in.bits.clear)
  val queuedContexts=WireDefault(0.U(16.W))
  val contextAvailable= !pending(io.in.bits.ctx) && !seedInFlight
  if(streaming) {
    // The mesh has a valid-only wave interface. TileSystem's cooldown
    // enforces the fixed commit interval before injection; an illegal wave
    // was already an assertion failure rather than a retryable handshake.
    operands.valid:=io.in.valid;io.in.ready:=operands.ready
    when(io.in.valid){assert(contextAvailable,"streaming wave reused an unfinished context or active seed")}
  }else if(cfg.stateReadStages==0) {
    operands.valid:=io.in.valid && contextAvailable
    io.in.ready:=operands.ready && contextAvailable
  }else {
    // The existing first preparation register buffers admission. Dependency
    // selection uses its registered context, keeping input ctx off ready.
    operands.valid:=io.in.valid;io.in.ready:=operands.ready
  }
  val raw=Wire(new Request)
  raw.a:=io.in.bits.a;raw.b:=io.in.bits.b
  raw.fmt:=io.in.bits.fmt;raw.op:=io.in.bits.op
  raw.mask:=io.in.bits.mask;raw.ctx:=io.in.bits.ctx;raw.tag:=io.in.bits.tag
  if(cfg.stages==2)raw.acc:=Mux(io.in.bits.clear || !initialized(io.in.bits.ctx),0.U,accum(io.in.bits.ctx))
  else raw.acc:=0.U // The registered local decoders below perform HF reads.
  operands.bits:=raw
  if(cfg.stateReadStages==0)alu.io.in<>operands
  else {
    // Reserve the downstream response before accepting a request. The first
    // preparation register can hold one dependent request; after it issues,
    // the arithmetic and commit stages advance without consumer backpressure.
    // The reservation includes both operand registers and the response queue.
    val request=Reg(new Request)
    val capture=WireDefault(true.B)
    val firstClear=Reg(Bool())
    val nextRequest=Mux(capture,operands.bits,request)
    val nextClear=Mux(capture,io.in.bits.clear,firstClear)
    request:=nextRequest;firstClear:=nextClear
    val advance=Wire(Bool())
    val unavailableForWords=WireDefault(0.U(16.W))
    val blockedForWords=WireDefault(false.B)
    val occupiedForWords=WireDefault(false.B)
    if(streaming) {
      operands.ready:=true.B;advance:=RegNext(operands.fire,false.B)
    }else {
      val occupied=RegInit(false.B)
      val previousPending=RegNext(pending.asUInt,0.U(16.W))
      val previousSeed=RegNext(seedInFlight,false.B)
      // Initialization snapshots follow a committing predecessor by one
      // clock. Keep that clock before consuming the buffered dependency.
      val unavailable=pending.asUInt | previousPending
      unavailableForWords:=unavailable;blockedForWords:=seedInFlight || previousSeed
      occupiedForWords:=occupied
      val available= !unavailable(request.ctx) && !seedInFlight && !previousSeed
      capture:= !occupied || available
      advance:=occupied && available
      queuedContexts:=Mux(occupied,UIntToOH(request.ctx,16),0.U)
      stateIssue:=advance;issueContext:=request.ctx;issueClear:=firstClear
      when(capture){occupied:=operands.fire}
      val capacity=cfg.peResponseCredits
      val credits=RegInit(capacity.U(log2Ceil(capacity+1).W))
      // Predict the next cycle's dependency admission at this register
      // boundary. previousPendingNext is pendingNow, so the OR with that
      // history cancels any same-clock commit clears. Only new issues can
      // add unavailable contexts. No response-credit or queue depth changes.
      val nextOccupied=Mux(capture,operands.fire,occupied)
      val nextUnavailable=pending.asUInt | Mux(advance,UIntToOH(request.ctx,16),0.U(16.W))
      val nextCapture= !nextOccupied ||
        (!nextUnavailable(nextRequest.ctx) && !io.seed.valid && !seedInFlight)
      val registeredCapture=RegNext(nextCapture,true.B)
      assert(registeredCapture===capture,"registered request-ready prediction diverged")
      operands.ready:=(credits=/=0.U || io.out.fire) && registeredCapture
      when(io.in.fire && !io.out.fire){credits:=credits-1.U}
      when(!io.in.fire && io.out.fire){credits:=credits+1.U}
      assert(credits<=capacity.U,"PE response reservation underflow")
    }
    val elasticWords=if(streaming)Seq.empty else Seq.tabulate(16) {word=>
      val bank=Module(new ElasticPreparationWord)
      bank.io.clock:=clock;bank.io.reset:=reset.asBool;bank.io.accepted:=operands.fire
      bank.io.unavailable:=unavailableForWords;bank.io.blocked:=blockedForWords
      bank.io.initialized:=initialized.asUInt;bank.io.ctx:=io.in.bits.ctx
      bank.io.fmt:=io.in.bits.fmt;bank.io.op:=io.in.bits.op;bank.io.clear:=io.in.bits.clear
      bank.io.data:=io.in.bits.asUInt.pad(512)(32*word+31,32*word)
      assert(bank.io.capture===capture,"local request capture diverged")
      assert(bank.io.state(5)===occupiedForWords,"local request occupancy diverged")
      when(occupiedForWords) {
        assert(bank.io.state(4,1)===request.ctx && bank.io.state(0)===firstClear,
          "local buffered request context diverged")
      }
      bank
    }
    val selected=if(streaming) {
      val copies=Module(new ContextReadReplicas)
      copies.io.clock:=clock;copies.io.ctx:=nextRequest.ctx
      copies.io.initialized:=initialized.asUInt;copies.io.clear:=nextClear
      copies.io.value
    }else Cat(elasticWords.reverse.map(_.io.selected))
    if(!streaming)when(advance) {
      val expected=Mux(firstClear || !initialized(request.ctx),0.U(16.W),UIntToOH(request.ctx,16))
      for(word<-0 until 16)assert(selected(16*word+15,16*word)===expected,
        "buffered admission consumed a stale initialization snapshot")
    }
    // The measured mesh critical path was one broadcast request_a bit to a
    // distant narrow-FP expansion slice. Capture local packed copies on the
    // SAME edge as request, before decoding, without adding a pipeline stage.
    val packedA=if(streaming) {
      val copies=Module(new PackedAReplicas);copies.io.clock:=clock;copies.io.data:=operands.bits.a(31,0)
      Some((0 until 16).map(lane=>copies.io.value(32*lane+31,32*lane)))
    }else None
    val preparation=if(streaming) {
      val controls=Module(new PreparationControlReplicas)
      controls.io.clock:=clock;controls.io.fmt:=nextRequest.fmt;controls.io.op:=nextRequest.op
      controls.io.value
    }else {
      Cat(elasticWords.reverse.map(_.io.controls))
    }
    val held=Wire(new Request);held:=request
    if(!streaming) {
      val payload=Cat(elasticWords.reverse.map(_.io.value))(io.in.bits.getWidth-1,0).asTypeOf(new PeRequest)
      held.a:=payload.a;held.b:=payload.b;held.fmt:=payload.fmt;held.op:=payload.op
      held.mask:=payload.mask;held.ctx:=payload.ctx;held.tag:=payload.tag
    }
    // Capture packed input encodings before narrow-FP expansion. This also
    // removes the generic PE's measured input-to-subnormal-decoder path.
    // Expansion uses local controls in the existing second operand stage;
    // context selection and metadata retain the same two-clock latency.
    val sources=ExpandSources(held,broadcastA=streaming,control=Some(preparation),packedA=packedA)
    val local=Wire(new Request);local:=held
    // Local encoded registers feed one decoder per 32-bit state slice.
    // Initialization/clear is selected before the register boundary.
    local.acc:=Cat((0 until 16).reverse.map {word=>
      Mux1H((0 until cfg.contexts).map(i=>selected(16*word+i)->accum(i)(32*word+31,32*word)))
    })
    val stateRead=RegNext(PrepareOperands(local,broadcastA=streaming,source=Some(sources),control=Some(preparation)))
    alu.io.in.valid:=RegNext(advance,false.B);alu.io.in.bits:=stateRead.request
    alu.io.floating.get:=stateRead.floating
    when(alu.io.in.valid){assert(alu.io.in.ready,"reserved PE operand could not advance")}
  }
  val commitWords=if(cfg.commitStages==0)Seq.empty else Seq.fill(16)(Module(new PeCommitWord(true)))
  val writeBack=Wire(new Response);val commitFire=Wire(Bool())
  if(cfg.commitStages==0) {
    io.out<>alu.io.out;writeBack:=io.out.bits;commitFire:=io.out.fire
  }
  else {
    // Commit state at the fixed arithmetic boundary. Response consumption
    // has its own reserved queue and cannot feed back into this write path.
    val occupied=commitWords.head.io.validOut
    val metadata=RegNext(alu.io.out.bits)
    val response=WireDefault(metadata)
    for((bank,word)<-commitWords.zipWithIndex) {
      bank.io.clock:=clock;bank.io.reset:=reset.asBool;bank.io.ready:=true.B
      bank.io.valid:=alu.io.out.valid;bank.io.ctx:=alu.io.out.bits.ctx
      bank.io.data:=alu.io.out.bits.data(32*word+31,32*word)
      assert(bank.io.validOut===occupied,"commit word validity diverged")
      when(occupied){assert(bank.io.address===commitWords.head.io.address,"commit word context diverged")}
    }
    response.data:=Cat(commitWords.reverse.map(_.io.value))
    writeBack:=response;commitFire:=occupied;alu.io.out.ready:=true.B
    if(streaming) {
      io.out.bits:=response;io.out.valid:=occupied
      when(io.out.valid){assert(io.out.ready,"streaming commit sink must accept every result")}
    }else {
      val capacity=cfg.peResponseCredits
      val stored=RegInit(0.U(log2Ceil(capacity+1).W))
      val enqueue=alu.io.out.valid;val dequeue=io.out.fire
      io.out.valid:=stored=/=0.U;responsesPending:=io.out.valid
      when(enqueue && !dequeue){stored:=stored+1.U}
      when(!enqueue && dequeue){stored:=stored-1.U}
      when(enqueue){assert(stored<capacity.U || dequeue,"reserved PE response slot was lost")}
      assert(stored<=capacity.U)
      val raw=alu.io.out.bits
      // Keep data words at natural 32-bit boundaries. The two final words
      // hold response metadata, not a second copy of the 512-bit payload.
      val meta=Cat(raw.tag,raw.ctx,raw.illegal,raw.overflow,raw.flags).pad(64)
      val packed=Cat(meta,raw.data)
      val words=(0 until 18).map {word=>
        val bank=Module(new ResponseDataBank(capacity));bank.io.clock:=clock;bank.io.reset:=reset.asBool
        bank.io.write:=alu.io.wordValid.get(word%16);bank.io.ready:=io.out.ready
        bank.io.data:=packed(32*word+31,32*word);bank.io.value
      }
      io.out.bits.data:=Cat(words.take(16).reverse)
      val head=Cat(words.drop(16).reverse)
      io.out.bits.flags:=head(4,0);io.out.bits.overflow:=head(20,5);io.out.bits.illegal:=head(21)
      io.out.bits.ctx:=head(25,22);io.out.bits.tag:=head(57,26)
    }
  }
  val writeContext=if(cfg.commitStages==0)UIntToOH(io.out.bits.ctx,16)
    else commitWords.head.io.address
  if(cfg.stages==2) {
    for(i<-0 until cfg.contexts) when(commitFire && writeContext(i)) {accum(i):=writeBack.data}
    when(io.seed.valid){accum(io.seed.bits.ctx):=io.seed.bits.data}
  }else {
    // Keep the commit and initialization gates beside each state word. A
    // single commit-valid FF must not drive all 8192 accumulator bit muxes.
    val next=Wire(Vec(cfg.contexts,Vec(16,UInt(32.W))))
    for(word<-0 until 16) {
      val local=Module(new StateWriteSelect)
      local.io.address:=commitWords(word).io.address
      local.io.commit:=commitWords(word).io.validOut
      for(i<-0 until cfg.contexts) {
        val value=Wire(UInt(32.W));value:=accum(i)(32*word+31,32*word)
        when(local.io.write(i)){value:=writeBack.data(32*word+31,32*word)}
        when(seedWords(word).io.selected(i)){value:=seedWords(word).io.value}
        next(i)(word):=value
      }
    }
    for(i<-0 until cfg.contexts)accum(i):=next(i).asUInt
  }
  for(i<-0 until cfg.contexts) when(commitFire && writeContext(i)) {
    flags(i):=flags(i)|writeBack.flags
    initialized(i):=true.B;pending(i):=false.B
  }
  // Preserve immediate clear on an uncontended admission. An operation
  // buffered behind its predecessor clears only when it actually issues.
  when(io.in.fire && io.in.bits.clear && !pending(io.in.bits.ctx) && !queuedContexts(io.in.bits.ctx)) {
    flags(io.in.bits.ctx):=0.U
  }
  when(stateIssue) {
    pending(issueContext):=true.B
    when(issueClear) { flags(issueContext):=0.U }
  }
  when(io.seed.valid) {
    assert(!pending.asUInt.orR && !queuedContexts.orR && !io.in.valid && !seedInFlight && !responsesPending,"context initialization requires a drained PE")
  }
  when(seedValid) {
    flags(seedContext):=seedFlags
    initialized(seedContext):=true.B
  }
  if(cfg.peReadLatency==0) {
    io.readData:=Mux(initialized(io.readCtx),accum(io.readCtx),0.U);io.readFlags:=flags(io.readCtx)
  }else {
    val address=RegEnable(UIntToOH(io.readCtx,16),0.U(16.W),io.readEnable)
    val selected=(0 until 16).map {word=>
      // Keep initialization beside each observed state word. These are
      // replicas of the live state, not a delayed observation snapshot:
      // simultaneous first commit/seed must keep its original read behavior.
      val local=Module(new ObservationState)
      local.io.clock:=clock;local.io.reset:=reset.asBool;local.io.address:=address
      local.io.commitAddress:=commitWords(word).io.address
      local.io.commit:=commitWords(word).io.validOut
      local.io.seedAddress:=seedWords(word).io.selected
      local.io.selected
    }
    io.readData:=RegNext(Cat((0 until 16).reverse.map {word=>
      Mux1H((0 until cfg.contexts).map(i=>selected(word)(i)->accum(i)(32*word+31,32*word)))
    }))
    io.readFlags:=RegNext(Mux1H((0 until cfg.contexts).map(i=>selected.head(i)->flags(i))))
  }
  io.pending:=pending.asUInt | queuedContexts | Mux(seedInFlight,"hffff".U(16.W),0.U(16.W))
}

class WaveToken extends Bundle {
  val data=UInt(64.W);val fmt=UInt(4.W);val ctx=UInt(4.W);val tag=UInt(32.W)
  val clear=Bool();val mask=UInt(16.W)
}
class WaveRegister extends BlackBox {
  override def desiredName="SimdWaveRegister"
  val io=IO(new Bundle {
    val clock=Input(Clock());val reset=Input(Bool());val valid=Input(Bool())
    val data=Input(UInt(121.W));val value=Output(UInt(121.W));val validOut=Output(Bool())
  })
}

/** Fully registered nearest-neighbor mesh. Admission is at tile boundaries;
  * a wave must only reuse a context after its feedback latency. */
class SystolicArray(cfg:Config) extends Module with RequireSyncReset {
  val io=IO(new Bundle {
    val inject=Input(Bool());val a=Input(Vec(cfg.rows,UInt(64.W)))
    val b=Input(Vec(cfg.cols,UInt(64.W)));val masks=Input(Vec(cfg.cols,UInt(16.W)))
    val rowMask=Input(UInt(cfg.rows.W));val fmt=Input(UInt(4.W));val ctx=Input(UInt(4.W))
    val tag=Input(UInt(32.W));val clear=Input(Bool())
    val readEnable=Input(Bool());val readCtx=Input(UInt(4.W));val readRow=Input(UInt(math.max(1,log2Ceil(cfg.rows)).W))
    val readCol=Input(UInt(math.max(1,log2Ceil(cfg.cols)).W));val readData=Output(UInt(512.W))
    val readFlags=Output(UInt(5.W));val busy=Output(Bool())
    val macs=Output(UInt(64.W));val completions=Output(UInt(64.W))
    val overflowEvents=Output(UInt(64.W));val exceptionOps=Output(UInt(64.W))
    val seed=Flipped(Valid(new ArraySeed))
  })
  def token(data:UInt,mask:UInt,valid:Bool):Valid[WaveToken] = {
    val x=Wire(Valid(new WaveToken));x.valid:=valid;x.bits.data:=data
    x.bits.mask:=mask;x.bits.fmt:=io.fmt;x.bits.ctx:=io.ctx
    x.bits.tag:=io.tag;x.bits.clear:=io.clear;x
  }
  // Track every register of a multi-clock hop. Otherwise busy can drop
  // between neighboring nodes while a masked or empty wave is still flying.
  val inFlight=scala.collection.mutable.ArrayBuffer[Bool]()
  def delay(x:Valid[WaveToken],n:Int):Valid[WaveToken] = (0 until n).foldLeft(x) { (a,_)=>
    val v=Wire(Valid(new WaveToken))
    if(cfg.stages==2){v.valid:=RegNext(a.valid,false.B);v.bits:=RegNext(a.bits)}
    else {
      // Identical diagonal metadata was being merged into shared FFs. Keep
      // each nearest-neighbor bank distinct so physical control follows data.
      require((new WaveToken).getWidth==121)
      val bank=Module(new WaveRegister);bank.io.clock:=clock;bank.io.reset:=reset.asBool
      bank.io.valid:=a.valid;bank.io.data:=a.bits.asUInt
      v.valid:=bank.io.validOut;v.bits:=bank.io.value.asTypeOf(new WaveToken)
    }
    inFlight+=v.valid
    v
  }
  val horizontal=Array.ofDim[Valid[WaveToken]](cfg.rows,cfg.cols+1)
  val vertical=Array.ofDim[Valid[WaveToken]](cfg.rows+1,cfg.cols)
  for(r<-0 until cfg.rows) horizontal(r)(0)=delay(token(io.a(r),0xffff.U,io.inject&&io.rowMask(r)),r*cfg.hop)
  for(c<-0 until cfg.cols) vertical(0)(c)=delay(token(io.b(c),io.masks(c),io.inject),c*cfg.hop)
  val pes=Seq.tabulate(cfg.rows,cfg.cols)((_,_)=>Module(new SimdPE(cfg,streaming=true)))
  val accepted=scala.collection.mutable.ArrayBuffer[UInt]()
  val issues=scala.collection.mutable.ArrayBuffer[Bool]()
  for(r<-0 until cfg.rows;c<-0 until cfg.cols) {
    val a=horizontal(r)(c);val b=vertical(r)(c);val p=pes(r)(c)
    p.io.seed.valid:=io.seed.valid && io.seed.bits.row===r.U && io.seed.bits.col===c.U
    p.io.seed.bits:=io.seed.bits.value
    p.io.in.valid:=a.valid&&b.valid
    // A is a scalar along K; broadcast inside each fixed-width arithmetic
    // branch, avoiding a format-controlled packed broadcast before expansion.
    p.io.in.bits.a:=a.bits.data;p.io.in.bits.b:=b.bits.data;p.io.in.bits.fmt:=a.bits.fmt
    val effectiveMask=b.bits.mask & Formats.mask(a.bits.fmt)
    p.io.in.bits.mask:=effectiveMask;p.io.in.bits.ctx:=a.bits.ctx;p.io.in.bits.tag:=a.bits.tag
    p.io.in.bits.clear:=a.bits.clear;p.io.in.bits.op:=1.U;p.io.out.ready:=true.B;p.io.readCtx:=io.readCtx
    p.io.readEnable:=io.readEnable && io.readRow===r.U && io.readCol===c.U
    when(p.io.in.valid) {
      assert(p.io.in.ready,"wave reuses an unfinished context")
      assert(a.bits.ctx===b.bits.ctx && a.bits.tag===b.bits.tag,"systolic token misalignment")
    }
    horizontal(r)(c+1)=delay(a,cfg.hop);vertical(r+1)(c)=delay(b,cfg.hop)
    accepted+=Mux(p.io.in.fire,Formats.activeCount(b.bits.mask,a.bits.fmt,cfg.stages>2),0.U)
    issues+=p.io.in.fire
  }
  when(io.seed.valid){assert(!io.busy && !io.inject && io.seed.bits.row<cfg.rows.U && io.seed.bits.col<cfg.cols.U)}
  val data=VecInit(pes.flatten.map(_.io.readData));val fl=VecInit(pes.flatten.map(_.io.readFlags))
  val index=(io.readRow*cfg.cols.U+io.readCol)(math.max(1,log2Ceil(cfg.rows*cfg.cols))-1,0)
  if(cfg.stages==2) {
    io.readData:=(if(cfg.rows*cfg.cols==1)data(0) else data(index))
    io.readFlags:=(if(cfg.rows*cfg.cols==1)fl(0) else fl(index))
  }else {
    // One registered 2x2 choice per level; a read crosses only neighboring
    // groups at each clock. Address bits follow the same pipeline as data.
    val levels=log2Ceil(math.max(cfg.rows,cfg.cols));val size=1<<levels
    var groups=Seq.tabulate(size,size) { (r,c)=>
      if(r<cfg.rows&&c<cfg.cols)(pes(r)(c).io.readData,pes(r)(c).io.readFlags)
      else (0.U(512.W),0.U(5.W))
    }
    for(level<-0 until levels) {
      val delay=cfg.peReadLatency+cfg.hop*level+cfg.hop-1
      val row=Pipes.delay(io.readRow.pad(levels)(level),delay,true.B)
      val col=Pipes.delay(io.readCol.pad(levels)(level),delay,true.B)
      val select=Pipes.delay(Cat(io.readRow.pad(levels)(level),io.readCol.pad(levels)(level)),delay-1,true.B)
      val previous=groups
      groups=Seq.tabulate(previous.length/2,previous.length/2) { (r,c)=>
        val relayed=Seq.tabulate(2,2) { (rr,cc)=>
          val source=previous(2*r+rr)(2*c+cc)
          (Pipes.delay(source._1,cfg.hop-1,true.B),Pipes.delay(source._2,cfg.hop-1,true.B))
        }
        val local=Module(new ReadControlReplicas);local.io.clock:=clock;local.io.data:=select
        def choose(field:((UInt,UInt))=>UInt,rr:Bool,cc:Bool):UInt=Mux(rr,
          Mux(cc,field(relayed(1)(1)),field(relayed(1)(0))),
          Mux(cc,field(relayed(0)(1)),field(relayed(0)(0))))
        val words=(0 until 16).map {word=>
          RegNext(choose(x=>x._1(32*word+31,32*word),local.io.value(2*word+1),local.io.value(2*word)))
        }
        (Cat(words.reverse),RegNext(choose(_._2,row,col)))
      }
    }
    io.readData:=groups.head.head._1;io.readFlags:=groups.head.head._2
  }
  val waveBusy=horizontal.flatten.filter(_!=null).map(_.valid)++vertical.flatten.filter(_!=null).map(_.valid)++inFlight
  def treeSum(xs:Seq[UInt]):UInt = if(xs.length==1)xs.head else treeSum(xs.grouped(2).map {
    case Seq(a,b)=>a+&b
    case Seq(a)=>a
  }.toSeq)
  def counter(xs:Seq[UInt]):UInt = {
    val width=math.max(1,log2Ceil(cfg.rows*cfg.cols*16+1))
    val local=xs.grouped(cfg.cols).map {x=>
      val inputs=x.map {v=>val n=Wire(UInt(width.W));n:=v;n}
      val reg=RegInit(0.U(width.W));reg:=treeSum(inputs);reg
    }.toSeq
    val reduced=RegInit(0.U(width.W));reduced:=treeSum(local)
    val total=RegInit(0.U(64.W));total:=(if(cfg.stages==2)total+reduced else FastCarry.add(total,reduced.pad(64),64));total
  }
  def event(x:UInt):UInt=if(cfg.stages==2)x else {val r=RegInit(0.U(x.getWidth.W));r:=x;r}
  val commits=pes.flatten.map(p=>event(p.io.out.fire.asUInt))
  // Count per-PE lane events before the event register. Previously the
  // following row stage contained both sixteen-bit popcounts and a row sum;
  // that combined path failed the 1 ns CTS check. Registering each count
  // keeps the same event latency and separates the two reductions.
  val overflow=pes.flatten.map {p=>
    // A routed critical path launched from commit-valid, traversed the
    // entire population tree and ended at this event register. Count the
    // payload independently; valid then gates five result bits at its end.
    val count=if(cfg.stages==2) {
      LanePopulation.count(Mux(p.io.out.fire,p.io.out.bits.overflow,0.U(16.W)),false)
    }else Mux(p.io.out.fire,LanePopulation.count(p.io.out.bits.overflow,true),0.U(5.W))
    event(count)
  }
  val exceptions=pes.flatten.map(p=>event((p.io.out.fire&&p.io.out.bits.flags.orR).asUInt))
  val activity=issues.reduce(_||_)||pes.flatten.map(_.io.out.fire).reduce(_||_)
  val countPending1=RegNext(activity,false.B);val countPending2=RegNext(countPending1,false.B)
  val countPending3=if(cfg.stages==2)false.B else RegNext(countPending2,false.B)
  val rawBusy=pes.flatten.map(_.io.pending.orR).reduce(_||_) || waveBusy.reduce(_||_) || countPending1||countPending2||countPending3
  // Keep the array-wide reduction away from the output timing path. An
  // injection is visible immediately and is included in rawBusy before its
  // first edge, so registering release cannot expose an idle gap.
  io.busy:=(if(cfg.busyReleaseLatency==0)rawBusy else RegNext(rawBusy,false.B) || io.inject || RegNext(io.seed.valid,false.B))
  io.macs:=counter(accepted.toSeq.map(event(_)));io.completions:=counter(commits)
  io.overflowEvents:=counter(overflow);io.exceptionOps:=counter(exceptions)
}
