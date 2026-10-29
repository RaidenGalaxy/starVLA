package simd

import chisel3._
import chisel3.util._

/** Row-major source tensors. Rows are byte addressed and padded to 8 bytes;
  * INT4 stores the even element in the low nibble. C uses the native wide
  * accumulator element format (little endian, no narrowing). */
class GemmDescriptor extends Bundle {
  val a=UInt(64.W);val b=UInt(64.W);val c=UInt(64.W)
  val aStride=UInt(64.W);val bStride=UInt(64.W);val cStride=UInt(64.W)
  val m=UInt(32.W);val n=UInt(32.W);val k=UInt(32.W)
  val fmt=UInt(4.W);val weight4=Bool();val accumulate=Bool();val tag=UInt(32.W)
  val weightScales=UInt(64.W);val scaleStride=UInt(64.W);val groupLog2=UInt(5.W)
  val grouped=Bool();val activationScales=UInt(64.W);val activationScaleStride=UInt(64.W)
  val prefetch=Bool()
  val outputConvert=Bool();val outputFormat=UInt(4.W);val outputScale=UInt(32.W);val outputZeroPoint=UInt(32.W)
}
object GemmPerformance {
  val names=Seq("address_pack_control","initial_c_seed","dram_request_wait","dram_response_wait",
    "tile_issue_wait","tile_completion_wait","result_readback","auxiliary_scale_convert","output_write")
}
class GemmCompletion extends Bundle {
  val tag=UInt(32.W);val error=Bool();val flags=UInt(5.W)
  val computeCycles=UInt(64.W);val macs=UInt(64.W);val overflow=UInt(64.W)
  val memoryCycles=UInt(64.W);val readBursts=UInt(64.W)
  val vectors=UInt(64.W);val feedbackStalls=UInt(64.W);val inputStalls=UInt(64.W)
  val dequantOperations=UInt(64.W);val dequantCycles=UInt(64.W)
  val stageCycles=Vec(GemmPerformance.names.length,UInt(64.W))
  val prefetchedTiles=UInt(64.W);val prefetchLoadBytes=UInt(64.W);val prefetchInFlightBytes=UInt(64.W)
  val epilogueElements=UInt(64.W);val epilogueAuxVectors=UInt(64.W);val epilogueCycles=UInt(64.W);val epilogueSaturations=UInt(64.W)
}
class NarrowFloat32 extends BlackBox {
  override def desiredName="SimdNarrow32"
  val io=IO(new Bundle {
    val value=Input(UInt(32.W));val bf16=Input(Bool());val result=Output(UInt(16.W));val flags=Output(UInt(5.W))
  })
}
class OutputConverter extends BlackBox {
  override def desiredName="SimdOutputConvert"
  val io=IO(new Bundle {
    val value=Input(UInt(64.W));val wide_source=Input(Bool());val format=Input(UInt(4.W));val zero_point=Input(UInt(32.W))
    val result=Output(UInt(64.W));val flags=Output(UInt(5.W));val saturated=Output(Bool())
  })
}

/** Conservative gather sequencer: one cached DRAM line at a time, sixteen
  * output contexts, K tiles of eight. All address generation, tails, packing,
  * context scheduling and output scatter are hardware operations. The cache
  * is deliberately simple; its stalls are part of measured end-to-end time. */
class GemmSequencer(cfg:Config) extends Module {
  require(isPow2(cfg.cols),"descriptor N tiling requires a power-of-two PE column count")
  require(cfg.rows*cfg.cols*cfg.contexts*8<=8192,"INT4 paired summaries must fit the aggregate state bank")
  val io=IO(new Bundle {
    val command=Flipped(Decoupled(new GemmDescriptor));val done=Decoupled(new GemmCompletion)
    val load=Decoupled(new LoadBeat);val tile=Decoupled(new TileCommand(cfg))
    val tileDone=Flipped(Decoupled(new TileDone));val read=Decoupled(new ReadCommand)
    val result=Flipped(Decoupled(new Response))
    val seed=Decoupled(new ArraySeed)
    val auxiliary=Decoupled(new AuxiliaryCommand)
    val outputStart=Decoupled(new OutputCommand);val outputPush=Decoupled(new OutputPushCommand)
    val outputDone=Flipped(Decoupled(new DmaCompletion))
    val memoryRequest=Decoupled(new MemoryReadRequest)
    val memoryResponse=Flipped(Decoupled(new MemoryReadResponse))
    val observed=Output(Valid(new Response))
    val inputTileCaptured=Input(Bool());val prefetchLoad=Output(Valid(UInt(5.W)))
  })
  private val states=Enum(38)
  val idle=states(0)
  val setup=states(1)
  val seedSelect=states(2)
  val seedByte=states(3)
  val seedSend=states(4)
  val element=states(5)
  val scale=states(6)
  val dequantSend=states(7)
  val dequantWait=states(8)
  val narrow=states(9)
  val dequantPack=states(10)
  val missRequest=states(11)
  val missResponse=states(12)
  val store=states(13)
  val launch=states(14)
  val compute=states(15)
  val select=states(16)
  val read=states(17)
  val result=states(18)
  val startWrite=states(19)
  val pushWrite=states(20)
  val waitWrite=states(21)
  val complete=states(22)
  val groupScaleA=states(23);val groupScaleB=states(24)
  val groupGammaSend=states(25);val groupGammaWait=states(26)
  val groupConvertSend=states(27);val groupConvertWait=states(28)
  val groupFmaSend=states(29);val groupFmaWait=states(30)
  val epiPrepare=states(31);val epiWide=states(32);val epiIntegerSend=states(33);val epiIntegerWait=states(34)
  val epiScaleSend=states(35);val epiScaleWait=states(36);val epiConvert=states(37)
  val state=RegInit(idle);val desc=Reg(new GemmDescriptor)
  val mi=RegInit(0.U(32.W));val nj=RegInit(0.U(32.W));val kb=RegInit(0.U(32.W))
  val contexts=Reg(UInt(5.W));val kt=Reg(UInt(4.W));val ctx=RegInit(0.U(4.W))
  val kk=RegInit(0.U(3.W));val word=RegInit(0.U(4.W));val lane=RegInit(0.U(4.W))
  val bank=RegInit(false.B);val packed=RegInit(0.U(64.W))
  val tag=RegInit(0.U(32.W));val responseTag=Reg(UInt(32.W))
  val flags=RegInit(0.U(5.W));val failed=RegInit(false.B)
  val cycles=RegInit(0.U(64.W));val computeCycles=RegInit(0.U(64.W))
  val macs=RegInit(0.U(64.W));val overflow=RegInit(0.U(64.W));val bursts=RegInit(0.U(64.W))
  val vectors=RegInit(0.U(64.W));val feedbackStalls=RegInit(0.U(64.W));val inputStalls=RegInit(0.U(64.W))
  val dequantOperations=RegInit(0.U(64.W));val dequantCycles=RegInit(0.U(64.W));val dequantPending=RegInit(false.B)
  val tileRunning=RegInit(false.B);val prefetching=RegInit(false.B);val activeTileTag=Reg(UInt(32.W))
  val prefetchedTiles=RegInit(0.U(64.W));val prefetchLoadBytes=RegInit(0.U(64.W));val prefetchInFlightBytes=RegInit(0.U(64.W))
  val epiSource=Reg(UInt(512.W));val epiScaled=Reg(UInt(64.W));val epiOutput=Reg(UInt(512.W))
  val epilogueElements=RegInit(0.U(64.W));val epilogueAuxVectors=RegInit(0.U(64.W))
  val epilogueCycles=RegInit(0.U(64.W));val epilogueSaturations=RegInit(0.U(64.W))
  val nibbleValid=RegInit(VecInit(Seq.fill(cfg.rows)(false.B)))
  val nibbleData=Reg(Vec(cfg.rows,UInt(4.W)));val nibbleAddress=Reg(Vec(cfg.rows,UInt(64.W)))
  val stageCycles=RegInit(VecInit(Seq.fill(GemmPerformance.names.length)(0.U(64.W))))
  def inStates(values:UInt*):Bool=values.map(state===_).reduce(_||_)
  val epilogueActive=inStates(epiPrepare,epiWide,epiIntegerSend,epiIntegerWait,epiScaleSend,epiScaleWait,epiConvert)
  when(epilogueActive){epilogueCycles:=epilogueCycles+1.U}
  val stages=Seq(inStates(setup,element,store,select),inStates(seedSelect,seedByte,seedSend),
    state===missRequest,state===missResponse,state===launch,state===compute,inStates(read,result),
    inStates(scale,dequantSend,dequantWait,narrow,dequantPack,groupScaleA,groupScaleB,
      groupGammaSend,groupGammaWait,groupConvertSend,groupConvertWait,groupFmaSend,groupFmaWait)||epilogueActive,
    inStates(startWrite,pushWrite,waitWrite))
  for(i<-stages.indices)when(stages(i)){stageCycles(i):=stageCycles(i)+1.U}
  when(state=/=idle && state=/=complete){assert(PopCount(VecInit(stages))===1.U,"sequencer phases must partition every active memory clock")}
  when(dequantPending){dequantCycles:=dequantCycles+1.U}
  when(state=/=idle && state=/=complete){cycles:=cycles+1.U}
  val width=Formats.width(desc.fmt);val lanes=Formats.lanes(desc.fmt)
  val floatingWeight4=desc.weight4 && (desc.fmt===7.U || desc.fmt===8.U)
  val grouped=desc.grouped
  val groupSize=1.U(32.W)<<desc.groupLog2
  val groupRemaining=groupSize-(kb & (groupSize-1.U))
  val lastGroup=kb+kt>=desc.k
  val groupBoundary=(kb+kt)>=desc.k || ((kb+kt)&(groupSize-1.U))===0.U
  val laneShift=6.U-MuxLookup(desc.fmt,2.U)(Formats.widths.zipWithIndex.map{case(w,f)=>f.U->log2Ceil(w).U})
  val spanShift=laneShift+log2Ceil(cfg.cols).U
  val span=(1.U(32.W)<<spanShift)(31,0)
  val accBytes=Mux(grouped,4.U,VecInit(Formats.accs.map(w=>(w/8).U(5.W)))(desc.fmt))
  val row=RegInit(0.U(4.W));val col=RegInit(0.U(4.W))
  val outputBase=nj+(ctx*span)+col*lanes
  val outputColumn=outputBase+Mux(grouped||desc.outputConvert,lane,0.U)
  val outputCount=Mux(desc.n-outputBase<lanes,desc.n-outputBase,lanes)
  val groupPairCount=Mux(outputCount-lane>=2.U,2.U,1.U)
  val groupMask=Mux(groupPairCount===2.U,3.U(16.W),1.U(16.W))
  val outputWidth=Formats.width(desc.outputFormat)
  val outputBytes=Mux(desc.outputConvert,(groupPairCount*outputWidth+7.U)>>3,
    Mux(grouped,groupPairCount<<2,(outputCount*accBytes)(6,0)))
  val cAddress=desc.c+(mi+row)*desc.cStride+Mux(desc.outputConvert,(outputColumn*outputWidth)>>3,outputColumn*accBytes)
  val seedData=Reg(UInt(512.W));val seedOffset=Reg(UInt(6.W));val seedMiss=RegInit(false.B)

  val cache=Reg(Vec(4,UInt(128.W)));val cacheTag=Reg(UInt(58.W));val cacheValid=RegInit(false.B)
  val fillAddress=Reg(UInt(64.W));val received=RegInit(0.U(4.W))
  val scaleCache=Reg(Vec(4,UInt(128.W)));val scaleTag=Reg(UInt(58.W));val scaleValid=RegInit(false.B)
  // Alternating A/B scale accesses must not evict each other for every
  // output. These independent lines remain coherent within one descriptor;
  // both are invalidated at the next command boundary.
  val activationScaleCache=Reg(Vec(4,UInt(128.W)))
  val activationScaleTag=Reg(UInt(58.W));val activationScaleValid=RegInit(false.B)
  val activationScaleMiss=RegInit(false.B)
  val scaleMiss=RegInit(false.B);val code=Reg(UInt(4.W));val scaleValue=Reg(UInt(32.W))
  val dequantValue=Reg(UInt(32.W));val narrowed=Reg(UInt(16.W))
  val groupPartials=Reg(UInt(512.W));val groupActivationScale=Reg(UInt(32.W));val groupWeightScale=Reg(UInt(64.W))
  val groupGamma=Reg(UInt(64.W));val groupConverted=Reg(UInt(64.W))
  val scaleResume=Reg(UInt(5.W))
  val sourceRow=Mux(bank,kb+kk,mi+word)
  val sourceColumn=Mux(bank,nj+ctx*span+word*lanes+lane,kb+kk)
  val groupIndex=kb>>desc.groupLog2
  val scaleAddress=Mux(state===groupScaleA,
    desc.activationScales+groupIndex*desc.activationScaleStride+(mi+row)*4.U,
    Mux(state===groupScaleB,desc.weightScales+groupIndex*desc.scaleStride+outputColumn*4.U,
      desc.weightScales+((kb+kk)>>desc.groupLog2)*desc.scaleStride+sourceColumn*4.U))
  val activationScale=state===groupScaleA
  val scaleHit=Mux(activationScale,activationScaleValid && activationScaleTag===scaleAddress(63,6),
    scaleValid && scaleTag===scaleAddress(63,6))
  val scaleLine=Mux(activationScale,activationScaleCache.asUInt,scaleCache.asUInt)
  val sourceWidth=Mux(bank&&desc.weight4,4.U,width)
  val sourceBits=sourceColumn*sourceWidth
  val sourceAddress=Mux(state===seedByte,cAddress+seedOffset,
    Mux(bank,desc.b,desc.a)+sourceRow*Mux(bank,desc.bStride,desc.aStride)+(sourceBits>>3))
  val active=Mux(bank,sourceColumn<desc.n,sourceRow<desc.m)
  val hit=cacheValid && cacheTag===sourceAddress(63,6)
  val shifted=cache.asUInt >> Cat(sourceAddress(5,0),sourceBits(2,0))
  val elementMask=((1.U(65.W)<<sourceWidth)-1.U)(63,0)
  val raw=shifted(63,0)&elementMask
  val expanded=Mux(bank&&desc.weight4,Cat(Fill(4,raw(3)),raw(3,0)).pad(64),raw)
  val value=Mux(active,expanded,0.U)
  val nextPacked=packed | (value<<(lane*width))(63,0)

  io.command.ready:=state===idle
  when(io.command.fire) {
    stageCycles.foreach(_:=0.U)
    val d=io.command.bits
    assert(d.fmt<=10.U && d.m=/=0.U && d.n=/=0.U && d.k>=1.U && d.k<=65536.U)
    assert(!d.weight4 || d.fmt===1.U || d.fmt===7.U || d.fmt===8.U,"mixed weights require INT8, FP16 or BF16 activations")
    when(d.grouped) {
      assert((d.fmt===0.U || d.fmt===1.U || d.fmt===5.U || d.fmt===6.U) && !d.accumulate,
        "group scaling uses INT4/INT8/FP4/FP8 inputs and a zero FP32 initial C")
      assert(d.groupLog2<=16.U && d.weightScales(2,0)===0.U && d.activationScales(2,0)===0.U)
      assert(d.scaleStride(2,0)===0.U && d.scaleStride>=d.n*4.U)
      assert(d.activationScaleStride(2,0)===0.U && d.activationScaleStride>=d.m*4.U)
    }
    when(d.weight4 && d.fmt=/=1.U) {
      assert(d.groupLog2<=16.U && d.weightScales(2,0)===0.U && d.scaleStride(2,0)===0.U && d.scaleStride>=d.n*4.U)
    }
    assert(d.a(2,0)===0.U && d.b(2,0)===0.U && d.aStride(2,0)===0.U && d.bStride(2,0)===0.U,
      "source bases and row strides must be multiples of eight bytes")
    val w=Formats.width(d.fmt)
    val bw=Mux(d.weight4,4.U,w)
    assert(d.aStride>=((d.k*w+7.U)>>3) && d.bStride>=((d.n*bw+7.U)>>3))
    val ab=Mux(d.grouped,4.U,VecInit(Formats.accs.map(w=>(w/8).U(5.W)))(d.fmt))
    assert(d.cStride>=Mux(d.outputConvert,(d.n*Formats.width(d.outputFormat)+7.U)>>3,d.n*ab))
    when(d.outputConvert){
      assert(d.outputFormat<=10.U,"unsupported output format")
      assert(!d.accumulate,"converted output currently requires zero initial C")
    }
    desc:=d;mi:=0.U;nj:=0.U;kb:=0.U;state:=setup
    flags:=0.U;failed:=false.B;cycles:=0.U;computeCycles:=0.U;macs:=0.U;overflow:=0.U;bursts:=0.U
    vectors:=0.U;feedbackStalls:=0.U;inputStalls:=0.U
    dequantOperations:=0.U;dequantCycles:=0.U;dequantPending:=false.B;scaleValid:=false.B;scaleMiss:=false.B;seedMiss:=false.B
    activationScaleValid:=false.B;activationScaleMiss:=false.B
    cacheValid:=false.B;tag:=0.U
    tileRunning:=false.B;prefetching:=false.B
    prefetchedTiles:=0.U;prefetchLoadBytes:=0.U;prefetchInFlightBytes:=0.U
    epilogueElements:=0.U;epilogueAuxVectors:=0.U;epilogueCycles:=0.U;epilogueSaturations:=0.U
    nibbleValid.foreach(_:=false.B)
  }
  when(state===setup) {
    val full=(desc.n-nj)>>spanShift
    contexts:=Mux(full===0.U,1.U,Mux(full>16.U,16.U,full))
    val remaining=Mux(grouped && groupRemaining<desc.k-kb,groupRemaining,desc.k-kb)
    kt:=Mux(remaining>8.U,8.U,remaining)
    bank:=false.B;kk:=0.U;ctx:=0.U;word:=0.U;lane:=0.U;packed:=0.U
    row:=0.U;col:=0.U;state:=Mux(kb===0.U && desc.accumulate,seedSelect,element)
  }
  def nextSeed():Unit = {
    state:=seedSelect
    when(col===(cfg.cols-1).U){col:=0.U
      when(row===(cfg.rows-1).U){row:=0.U
        when(ctx===contexts-1.U){ctx:=0.U;state:=element}.otherwise{ctx:=ctx+1.U}
      }.otherwise{row:=row+1.U}
    }.otherwise{col:=col+1.U}
  }
  when(state===seedSelect) {
    when(mi+row>=desc.m || outputColumn>=desc.n){nextSeed()}
    .otherwise{seedData:=0.U;seedOffset:=0.U;state:=seedByte}
  }
  when(state===seedByte) {
    when(hit) {
      val byte=(cache.asUInt>>Cat(sourceAddress(5,0),0.U(3.W)))(7,0)
      seedData:=seedData|(byte.pad(512)<<(seedOffset<<3))(511,0)
      when(seedOffset+&1.U===outputBytes){state:=seedSend}.otherwise{seedOffset:=seedOffset+1.U}
    }.otherwise{
      fillAddress:=Cat(sourceAddress(63,6),0.U(6.W));received:=0.U
      cacheValid:=false.B;seedMiss:=true.B;scaleMiss:=false.B;state:=missRequest
    }
  }
  io.seed.valid:=state===seedSend;io.seed.bits.row:=row;io.seed.bits.col:=col
  io.seed.bits.value.ctx:=ctx;io.seed.bits.value.data:=seedData;io.seed.bits.value.flags:=0.U
  when(io.seed.fire){nextSeed()}
  when(state===element) {
    when(!active || hit) {
      val packedSource=(cache.asUInt>>Cat(sourceAddress(5,0),0.U(3.W)))(63,0)
      val extendedInt4=Cat((0 until 8).reverse.map {i=>
        val nibble=packedSource(4*i+3,4*i);Cat(Fill(4,nibble(3)),nibble)
      })
      // A full SIMD B word is contiguous in the row-major source. Bypass
      // per-element assembly only when every lane and source byte is valid.
      // Tail words and W4A16 retain their original scalar fallback.
      val sourceBytes=Mux(desc.weight4,4.U,8.U)
      val wholeWord=active && hit && bank && lane===0.U && !floatingWeight4 &&
        sourceColumn+&lanes<=desc.n && sourceBits(2,0)===0.U && sourceAddress(5,0)<=64.U-sourceBytes
      when(wholeWord) {
        packed:=Mux(desc.weight4,extendedInt4,packedSource);state:=store
      }.elsewhen(active && bank && floatingWeight4) {
        code:=raw(3,0);dequantPending:=true.B;state:=scale
      }.otherwise{
        packed:=nextPacked
        when(!bank || lane===lanes-1.U){state:=store}.otherwise{lane:=lane+1.U}
      }
    }.otherwise{
      fillAddress:=Cat(sourceAddress(63,6),0.U(6.W));received:=0.U;cacheValid:=false.B;seedMiss:=false.B;scaleMiss:=false.B;state:=missRequest
    }
  }
  when(state===scale) {
    when(scaleHit) {
      scaleValue:=(scaleLine>>Cat(scaleAddress(5,0),0.U(3.W)))(31,0);state:=dequantSend
    }.otherwise{
      fillAddress:=Cat(scaleAddress(63,6),0.U(6.W));received:=0.U
      scaleValid:=false.B;activationScaleMiss:=false.B;seedMiss:=false.B;scaleMiss:=true.B;scaleResume:=scale;state:=missRequest
    }
  }
  val int4ToFp32=VecInit((0 until 16).map {encoded=>
    val value=if(encoded<8)encoded else encoded-16
    (java.lang.Float.floatToRawIntBits(value.toFloat).toLong & 0xffffffffL).U(32.W)
  })
  for((stage,next,activation)<-Seq((groupScaleA,groupScaleB,true),(groupScaleB,groupGammaSend,false))) {
    when(state===stage) {
      when(scaleHit) {
        val value=scaleLine>>Cat(scaleAddress(5,0),0.U(3.W))
        if(activation)groupActivationScale:=value(31,0) else groupWeightScale:=value(63,0)
        state:=next
      }.otherwise {
        fillAddress:=Cat(scaleAddress(63,6),0.U(6.W));received:=0.U
        if(activation)activationScaleValid:=false.B else scaleValid:=false.B
        activationScaleMiss:=activation.B
        seedMiss:=false.B;scaleMiss:=true.B;scaleResume:=stage;state:=missRequest
      }
    }
  }
  io.auxiliary.valid:=state===dequantSend || state===groupGammaSend || state===groupConvertSend || state===groupFmaSend ||
    state===epiIntegerSend || state===epiScaleSend
  io.auxiliary.bits:=0.U.asTypeOf(new AuxiliaryCommand)
  io.auxiliary.bits.request.a:=int4ToFp32(code);io.auxiliary.bits.request.b:=scaleValue
  io.auxiliary.bits.request.fmt:=9.U;io.auxiliary.bits.request.op:=0.U
  io.auxiliary.bits.request.mask:=1.U;io.auxiliary.bits.request.tag:=tag
  when(state===groupGammaSend) {
    io.auxiliary.bits.request.a:=Cat(groupActivationScale,groupActivationScale)
    io.auxiliary.bits.request.b:=groupWeightScale
  }
  when(state===groupGammaSend || state===groupConvertSend || state===groupFmaSend) {
    io.auxiliary.bits.request.mask:=groupMask
  }
  when(state===groupConvertSend) {
    io.auxiliary.bits.request.fmt:=desc.fmt;io.auxiliary.bits.request.op:=7.U
    io.auxiliary.bits.request.acc:=(groupPartials>>(lane<<5))(63,0)
  }
  when(state===groupFmaSend) {
    io.auxiliary.bits.request.a:=groupConverted;io.auxiliary.bits.request.b:=groupGamma
    io.auxiliary.bits.request.op:=1.U;io.auxiliary.bits.aggregate:=true.B
    io.auxiliary.bits.clearAggregate:=groupIndex===0.U
    val vectorIndex=ctx*(cfg.rows*cfg.cols).U+row*cfg.cols.U+col
    io.auxiliary.bits.aggregateIndex:=(vectorIndex*(lanes>>1)+(lane>>1)).pad(13)(12,0)
  }
  when(state===epiIntegerSend) {
    io.auxiliary.bits.request.fmt:=desc.fmt;io.auxiliary.bits.request.op:=7.U
    io.auxiliary.bits.request.acc:=epiSource
    io.auxiliary.bits.request.mask:=((1.U(17.W)<<outputCount(4,0))-1.U)(15,0)
  }
  when(state===epiScaleSend) {
    io.auxiliary.bits.request.a:=Mux(grouped,epiSource(63,0),(epiSource>>(lane<<5))(63,0))
    io.auxiliary.bits.request.b:=Cat(desc.outputScale,desc.outputScale)
    io.auxiliary.bits.request.mask:=groupMask
  }
  when(io.auxiliary.fire){
    responseTag:=tag;tag:=tag+1.U
    when(state===epiIntegerSend || state===epiScaleSend){epilogueAuxVectors:=epilogueAuxVectors+1.U}
      .otherwise{dequantOperations:=dequantOperations+1.U}
    state:=MuxLookup(state,dequantWait)(Seq(groupGammaSend->groupGammaWait,groupConvertSend->groupConvertWait,groupFmaSend->groupFmaWait,
      epiIntegerSend->epiIntegerWait,epiScaleSend->epiScaleWait))
  }
  val cast=Module(new NarrowFloat32);cast.io.value:=dequantValue;cast.io.bf16:=desc.fmt===8.U
  when(state===narrow){narrowed:=cast.io.result;flags:=flags|cast.io.flags;state:=dequantPack}
  when(state===dequantPack) {
    packed:=packed|(narrowed.pad(64)<<(lane*width))(63,0);dequantPending:=false.B
    when(lane===lanes-1.U){state:=store}.otherwise{lane:=lane+1.U;state:=element}
  }
  io.memoryRequest.valid:=state===missRequest
  io.memoryRequest.bits.address:=fillAddress;io.memoryRequest.bits.id:=0.U
  when(io.memoryRequest.fire){state:=missResponse;bursts:=bursts+1.U}
  io.memoryResponse.ready:=state===missResponse
  when(io.memoryResponse.fire) {
    val r=io.memoryResponse.bits
    assert(r.id===0.U && !received(r.beat),"unexpected or repeated gather response")
    when(scaleMiss){
      when(activationScaleMiss){activationScaleCache(r.beat):=r.data}
      .otherwise{scaleCache(r.beat):=r.data}
    }.otherwise{cache(r.beat):=r.data}
    received:=received|UIntToOH(r.beat,4)
    when(r.error){failed:=true.B}
    when((received|UIntToOH(r.beat,4)).andR){
      when(scaleMiss){
        when(activationScaleMiss){activationScaleTag:=fillAddress(63,6);activationScaleValid:=true.B}
        .otherwise{scaleTag:=fillAddress(63,6);scaleValid:=true.B}
        state:=scaleResume
      }
      .otherwise{cacheTag:=fillAddress(63,6);cacheValid:=true.B;state:=Mux(seedMiss,seedByte,element)}
    }
  }
  val slots=kk*contexts+ctx
  val bankWords=Mux(bank,((cfg.cols+1)/2).U,((cfg.rows+1)/2).U)
  io.load.valid:=state===store;io.load.bits.bank:=bank
  io.load.bits.address:=slots*bankWords+(word>>1)
  io.load.bits.data:=Mux(word(0),Cat(packed,0.U(64.W)),packed.pad(128))
  io.load.bits.strobe:=Mux(word(0),"hff00".U,"h00ff".U)
  when(io.load.fire) {
    packed:=0.U;lane:=0.U;state:=element
    when(word===Mux(bank,(cfg.cols-1).U,(cfg.rows-1).U)) {
      word:=0.U
      when(ctx===contexts-1.U) {
        ctx:=0.U
        when(kk===kt-1.U){kk:=0.U;when(bank){state:=launch}.otherwise{bank:=true.B}}
        .otherwise{kk:=kk+1.U}
      }.otherwise{ctx:=ctx+1.U}
    }.otherwise{word:=word+1.U}
  }
  io.prefetchLoad.valid:=io.load.fire && prefetching
  io.prefetchLoad.bits:=PopCount(io.load.bits.strobe)
  when(io.prefetchLoad.valid) {
    prefetchLoadBytes:=prefetchLoadBytes+io.prefetchLoad.bits
    when(tileRunning){prefetchInFlightBytes:=prefetchInFlightBytes+io.prefetchLoad.bits}
  }
  io.tile.valid:=state===launch && !tileRunning;io.tile.bits.fmt:=desc.fmt;io.tile.bits.contexts:=contexts
  io.tile.bits.k:=kt;io.tile.bits.clear:=Mux(grouped,(kb&(groupSize-1.U))===0.U,kb===0.U && !desc.accumulate);io.tile.bits.tag:=tag
  val rowCount=Mux(desc.m-mi<cfg.rows.U,desc.m-mi,cfg.rows.U)(3,0)
  io.tile.bits.rowMask:=((1.U((cfg.rows+1).W)<<rowCount)-1.U)(cfg.rows-1,0)
  for(c<-0 until cfg.cols) {
    val first=nj+c.U*lanes
    val count=Mux(first>=desc.n,0.U,Mux(desc.n-first<lanes,desc.n-first,lanes))
    io.tile.bits.masks(c):=((1.U(17.W)<<count(4,0))-1.U)(15,0)
  }
  when(io.tile.fire){
    state:=compute;tag:=tag+128.U;tileRunning:=true.B;prefetching:=false.B;activeTileTag:=io.tile.bits.tag
  }
  // Reuse SRAM only after TileSystem has copied the complete current tile
  // into its wave FIFO. Group boundaries retain their original read/scale
  // ordering. A pending completion is accepted even while packing, waiting
  // for DRAM, or waiting for W4A16's auxiliary conversion.
  val anotherKTile=kb+kt<desc.k && !(grouped && groupBoundary)
  val startPrefetch=state===compute && desc.prefetch && tileRunning && io.inputTileCaptured &&
    !io.tileDone.valid && anotherKTile
  when(startPrefetch){kb:=kb+kt;state:=setup;prefetching:=true.B;prefetchedTiles:=prefetchedTiles+1.U}
  io.tileDone.ready:=tileRunning
  when(io.tileDone.fire) {
    assert(io.tileDone.bits.tag===activeTileTag,"tile completion tag changed during prefetch")
    tileRunning:=false.B
    computeCycles:=computeCycles+io.tileDone.bits.cycles;macs:=macs+io.tileDone.bits.macs
    overflow:=overflow+io.tileDone.bits.overflow
    vectors:=vectors+io.tileDone.bits.vectors
    feedbackStalls:=feedbackStalls+io.tileDone.bits.feedbackStalls;inputStalls:=inputStalls+io.tileDone.bits.inputStalls
    when(!prefetching) {
      when(anotherKTile){kb:=kb+kt;state:=setup}
      .otherwise{ctx:=0.U;row:=0.U;col:=0.U;lane:=0.U;state:=select}
    }
  }
  def nextOutput():Unit = {
    state:=select
    when(col===(cfg.cols-1).U){col:=0.U
      when(row===(cfg.rows-1).U){row:=0.U
        when(ctx===contexts-1.U){
          ctx:=0.U;state:=setup
          when(grouped && !lastGroup){kb:=kb+kt}
          .otherwise {
            kb:=0.U
            when(nj+contexts*span>=desc.n){nj:=0.U
              when(mi+cfg.rows.U>=desc.m){state:=complete}.otherwise{mi:=mi+cfg.rows.U}
            }.otherwise{nj:=nj+contexts*span}
          }
        }.otherwise{ctx:=ctx+1.U}
      }.otherwise{row:=row+1.U}
    }.otherwise{col:=col+1.U}
  }
  when(state===select){when(mi+row>=desc.m || outputColumn>=desc.n){nextOutput()}.otherwise{state:=read}}
  io.read.valid:=state===read;io.read.bits.ctx:=ctx;io.read.bits.row:=row;io.read.bits.col:=col;io.read.bits.tag:=tag
  when(io.read.fire){responseTag:=tag;tag:=tag+1.U;state:=result}
  def nextGroupLane():Unit = {
    dequantPending:=false.B
    when(lane+&2.U>=outputCount){lane:=0.U;nextOutput()}
    .otherwise{lane:=lane+2.U;dequantPending:=true.B;state:=groupScaleA}
  }
  def nextConvertedLane():Unit = {
    when(lane+&2.U>=outputCount){lane:=0.U;nextOutput()}
    .otherwise{lane:=lane+2.U;state:=epiScaleSend}
  }
  val outputCast=Seq.fill(2)(Module(new OutputConverter))
  for((converter,i)<-outputCast.zipWithIndex) {
    converter.io.value:=(if(i==0)Mux(state===epiWide,epiSource(63,0),epiScaled(31,0).pad(64))else epiScaled(63,32).pad(64))
    converter.io.wide_source:=(i==0).B && state===epiWide
    converter.io.format:=desc.outputFormat;converter.io.zero_point:=desc.outputZeroPoint
  }
  when(state===epiPrepare){state:=Mux(desc.fmt<5.U,epiIntegerSend,Mux(desc.fmt===10.U,epiWide,epiScaleSend))}
  when(state===epiWide){epiSource:=outputCast.head.io.result(31,0);flags:=flags|outputCast.head.io.flags;state:=epiScaleSend}
  when(state===epiConvert) {
    val pair=groupPairCount===2.U
    flags:=flags|outputCast.head.io.flags|Mux(pair,outputCast(1).io.flags,0.U)
    epilogueElements:=epilogueElements+groupPairCount
    epilogueSaturations:=epilogueSaturations+outputCast.head.io.saturated+ (pair && outputCast(1).io.saturated)
    epiOutput:=Mux1H(Seq(4,8,16,32,64).map(w=>(outputWidth===w.U)->
      Cat(Mux(pair,outputCast(1).io.result(w-1,0),0.U(w.W)),outputCast.head.io.result(w-1,0)).pad(512)))
    state:=startWrite
    when(outputWidth===4.U && !pair) {
      when(!outputColumn(0) && outputColumn+1.U<desc.n) {
        nibbleValid(row):=true.B;nibbleData(row):=outputCast.head.io.result(3,0);nibbleAddress(row):=cAddress
        when(grouped){nextGroupLane()}.otherwise{nextConvertedLane()}
      }.elsewhen(outputColumn(0)) {
        assert(nibbleValid(row) && nibbleAddress(row)===cAddress,"packed output lost its preceding nibble")
        epiOutput:=Cat(outputCast.head.io.result(3,0),nibbleData(row));nibbleValid(row):=false.B
      }
    }
    when(outputWidth===4.U && pair){assert(!outputColumn(0),"packed output pair starts on a high nibble")}
  }
  io.result.ready:=state===result || state===dequantWait || state===groupGammaWait || state===groupConvertWait || state===groupFmaWait ||
    state===epiIntegerWait || state===epiScaleWait
  io.observed.valid:=io.result.fire && state===result;io.observed.bits:=io.result.bits
  when(io.result.fire){
    assert(io.result.bits.tag===responseTag)
    flags:=flags|io.result.bits.flags;when(io.result.bits.illegal){failed:=true.B}
    when(state===epiIntegerWait){epiSource:=io.result.bits.data;state:=epiScaleSend}
    .elsewhen(state===epiScaleWait){epiScaled:=io.result.bits.data(63,0);state:=epiConvert}
    .elsewhen(state===dequantWait){dequantValue:=io.result.bits.data(31,0);state:=narrow}
    .elsewhen(state===groupGammaWait){
      groupGamma:=io.result.bits.data(63,0)
      when(desc.fmt>=5.U){
        // Narrow floating products already accumulated into FP32, with one
        // RNE FMA per K element. Preserve those bits without a second cast.
        groupConverted:=(groupPartials>>(lane<<5))(63,0);state:=groupFmaSend
      }.otherwise{state:=groupConvertSend}
    }
    .elsewhen(state===groupConvertWait){groupConverted:=io.result.bits.data(63,0);state:=groupFmaSend}
    .elsewhen(state===groupFmaWait){
      when(lastGroup){
        dequantPending:=false.B
        when(desc.outputConvert){epiSource:=io.result.bits.data;state:=epiScaleSend}.otherwise{state:=startWrite}
      }.otherwise{nextGroupLane()}
    }
    .elsewhen(grouped){groupPartials:=io.result.bits.data;dequantPending:=true.B;state:=groupScaleA}
    .elsewhen(desc.outputConvert){epiSource:=io.result.bits.data;lane:=0.U;state:=epiPrepare}
    .otherwise{state:=startWrite}
  }
  io.outputStart.valid:=state===startWrite
  io.outputStart.bits.address:=cAddress;io.outputStart.bits.bytes:=outputBytes;io.outputStart.bits.tag:=responseTag
  when(io.outputStart.fire){state:=pushWrite}
  io.outputPush.valid:=state===pushWrite
  io.outputPush.bits.bytes:=outputBytes;io.outputPush.bits.resultTag:=responseTag
  io.outputPush.bits.overrideData:=desc.outputConvert;io.outputPush.bits.data:=epiOutput
  when(io.outputPush.fire){state:=waitWrite}
  io.outputDone.ready:=state===waitWrite
  when(io.outputDone.fire){
    when(io.outputDone.bits.error){failed:=true.B}
    when(grouped){nextGroupLane()}.elsewhen(desc.outputConvert){nextConvertedLane()}.otherwise{nextOutput()}
  }
  io.done.valid:=state===complete;io.done.bits.tag:=desc.tag;io.done.bits.error:=failed;io.done.bits.flags:=flags
  when(state===complete){
    assert(!tileRunning && !prefetching,"descriptor completed with a pending prefetched tile")
    assert(!nibbleValid.asUInt.orR,"descriptor completed with an incomplete packed output byte")
  }
  io.done.bits.computeCycles:=computeCycles;io.done.bits.macs:=macs;io.done.bits.overflow:=overflow
  io.done.bits.memoryCycles:=cycles;io.done.bits.readBursts:=bursts
  io.done.bits.vectors:=vectors;io.done.bits.feedbackStalls:=feedbackStalls;io.done.bits.inputStalls:=inputStalls
  io.done.bits.dequantOperations:=dequantOperations;io.done.bits.dequantCycles:=dequantCycles
  io.done.bits.stageCycles:=stageCycles
  io.done.bits.prefetchedTiles:=prefetchedTiles;io.done.bits.prefetchLoadBytes:=prefetchLoadBytes
  io.done.bits.prefetchInFlightBytes:=prefetchInFlightBytes
  io.done.bits.epilogueElements:=epilogueElements;io.done.bits.epilogueAuxVectors:=epilogueAuxVectors
  io.done.bits.epilogueCycles:=epilogueCycles;io.done.bits.epilogueSaturations:=epilogueSaturations
  when(io.done.fire){state:=idle}
}

/** Full GEMM descriptor entry point; shares exactly the TileSystem datapaths. */
class ScratchRequest extends Bundle {
  val bank=UInt(2.W);val address=UInt(13.W);val write=Bool()
  val data=UInt(128.W);val strobe=UInt(16.W);val tag=UInt(32.W)
}
class ScratchResponse extends Bundle {
  val data=UInt(128.W);val tag=UInt(32.W);val error=Bool()
}
class GemmSystem(cfg:Config) extends RawModule {
  val io=IO(new Bundle {
    val coreClock=Input(Clock());val memoryClock=Input(Clock())
    val coreReset=Input(AsyncReset());val memoryReset=Input(AsyncReset())
    val command=Flipped(Decoupled(new GemmDescriptor));val done=Decoupled(new GemmCompletion)
    val memoryRequest=Decoupled(new MemoryReadRequest);val memoryResponse=Flipped(Decoupled(new MemoryReadResponse))
    val writeRequest=Decoupled(new MemoryReadRequest);val writeData=Decoupled(new MemoryWriteData)
    val writeResponse=Flipped(Decoupled(new MemoryWriteResponse))
    val writeBursts=Output(UInt(64.W));val writeBytes=Output(UInt(64.W));val observed=Output(Valid(new Response))
    val sramReadBytes=Output(UInt(64.W));val sramWriteBytes=Output(UInt(64.W))
    val aggregateReadBytes=Output(UInt(64.W));val aggregateWriteBytes=Output(UInt(64.W))
    val coreActive=Output(Bool())
    val prefetchLoad=Output(Valid(UInt(5.W)))
    val scratch=Flipped(Decoupled(new ScratchRequest));val scratchResult=Decoupled(new ScratchResponse)
  })
  val tile=Module(new TileSystem(cfg))
  tile.io.coreClock:=io.coreClock;tile.io.memoryClock:=io.memoryClock
  tile.io.coreReset:=io.coreReset;tile.io.memoryReset:=io.memoryReset
  val resetSync=Module(new ResetSynchronizer)
  resetSync.io.clock:=io.memoryClock;resetSync.io.externalReset:=io.memoryReset.asBool || io.coreReset.asBool
  val localReset=resetSync.io.value.asAsyncReset
  val seq=withClockAndReset(io.memoryClock,localReset){Module(new GemmSequencer(cfg))}
  io.done<>seq.io.done;tile.io.command<>seq.io.tile
  seq.io.inputTileCaptured:=tile.io.inputTileCaptured;io.prefetchLoad:=seq.io.prefetchLoad
  withClockAndReset(io.memoryClock,localReset) {
    val response=Reg(new ScratchResponse);val responseValid=RegInit(false.B)
    val pending=RegInit(false.B);val pendingBank=Reg(UInt(2.W));val pendingWrite=Reg(Bool())
    val busy=responseValid || pending
    seq.io.command.valid:=io.command.valid && !busy;seq.io.command.bits:=io.command.bits
    io.command.ready:=seq.io.command.ready && !busy
    val allowed=seq.io.command.ready && !io.command.valid && !busy
    val x=io.scratch.bits;val aggregate=x.bank===3.U
    val legal=Mux(aggregate,!x.write || x.strobe(15,8)===0.U,
      x.address<Mux(x.bank===1.U,2048.U,1024.U))
    val load=allowed && io.scratch.valid && x.write && !aggregate && legal
    tile.io.load.valid:=seq.io.load.valid || load;tile.io.load.bits:=seq.io.load.bits
    when(load) {
      tile.io.load.bits.bank:=x.bank;tile.io.load.bits.address:=x.address(11,0)
      tile.io.load.bits.data:=x.data;tile.io.load.bits.strobe:=x.strobe
    }
    seq.io.load.ready:=tile.io.load.ready && !load
    tile.io.peek.valid:=allowed && io.scratch.valid && legal && !aggregate && !x.write
    tile.io.peek.bits.bank:=x.bank;tile.io.peek.bits.address:=x.address(11,0)
    tile.io.aggregateInspection.valid:=allowed && io.scratch.valid && legal && aggregate
    tile.io.aggregateInspection.bits.address:=x.address
    tile.io.aggregateInspection.bits.write:=x.write;tile.io.aggregateInspection.bits.data:=x.data(63,0)
    tile.io.aggregateInspection.bits.strobe:=x.strobe(7,0)
    io.scratch.ready:=allowed && (!legal || Mux(aggregate,tile.io.aggregateInspection.ready,
      Mux(x.write,tile.io.load.ready,tile.io.peek.ready)))
    tile.io.peekResult.ready:=pending && pendingBank=/=3.U && !pendingWrite && !responseValid
    tile.io.aggregateInspectionResult.ready:=pending && pendingBank===3.U && !responseValid
    io.scratchResult.valid:=responseValid;io.scratchResult.bits:=response
    when(io.scratchResult.fire){responseValid:=false.B}
    when(io.scratch.fire) {
      response.tag:=x.tag;response.error:= !legal;response.data:=0.U
      when(!legal){responseValid:=true.B}.otherwise{
        pending:=true.B;pendingBank:=x.bank;pendingWrite:=x.write
      }
    }
    when(pending && pendingBank=/=3.U && pendingWrite && tile.io.memoryIdle){
      pending:=false.B;responseValid:=true.B
    }
    when(tile.io.peekResult.fire){response.data:=tile.io.peekResult.bits;pending:=false.B;responseValid:=true.B}
    when(tile.io.aggregateInspectionResult.fire){response.data:=tile.io.aggregateInspectionResult.bits.pad(128);pending:=false.B;responseValid:=true.B}
  }

  seq.io.tileDone<>tile.io.done;tile.io.read<>seq.io.read;seq.io.result<>tile.io.result
  tile.io.seed<>seq.io.seed
  tile.io.outputStart<>seq.io.outputStart;tile.io.outputPush<>seq.io.outputPush;seq.io.outputDone<>tile.io.outputDone
  io.memoryRequest<>seq.io.memoryRequest;seq.io.memoryResponse<>io.memoryResponse;io.observed:=seq.io.observed
  io.writeRequest<>tile.io.writeRequest;io.writeData<>tile.io.writeData;tile.io.writeResponse<>io.writeResponse
  io.writeBursts:=tile.io.dmaWriteRequests;io.writeBytes:=tile.io.dmaWritePayloadBytes
  tile.io.dmaCommand.valid:=false.B;tile.io.dmaCommand.bits:=0.U.asTypeOf(new DmaCommand);tile.io.dmaDone.ready:=true.B
  tile.io.memoryRequest.ready:=false.B;tile.io.memoryResponse.valid:=false.B;tile.io.memoryResponse.bits:=0.U.asTypeOf(new MemoryReadResponse)
  tile.io.auxiliary<>seq.io.auxiliary
  io.sramReadBytes:=tile.io.sramReadBytes;io.sramWriteBytes:=tile.io.sramWriteBytes
  io.aggregateReadBytes:=tile.io.aggregateReadBytes;io.aggregateWriteBytes:=tile.io.aggregateWriteBytes
  io.coreActive:=tile.io.coreActive
}
