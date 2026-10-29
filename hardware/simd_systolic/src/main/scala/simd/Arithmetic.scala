package simd

import chisel3._
import chisel3.util._

object Formats {
  val widths=Seq(4,8,16,32,64,4,8,16,16,32,64)
  val accs=Seq(32,32,48,80,144,32,32,32,32,32,64)
  def width(f:UInt):UInt = VecInit(widths.map(_.U(7.W)))(f)
  def lanes(f:UInt):UInt = VecInit(widths.map(w=>(64/w).U(5.W)))(f)
  private def widthMatches(f:UInt,w:Int):Bool=widths.zipWithIndex.collect{case(width,index)if width==w=>f===index.U}.reduce(_||_)
  def mask(f:UInt):UInt = {
    val s4=widthMatches(f,4);val s8=s4||widthMatches(f,8)
    val s16=s8||widthMatches(f,16);val s32=s16||widthMatches(f,32)
    val s64=s32||widthMatches(f,64)
    Cat(Fill(8,s4),Fill(4,s8),Fill(2,s16),s32,s64)
  }
  def activeCount(bits:UInt,f:UInt,preserveTree:Boolean=false):UInt = {
    // Count each supported prefix before the precision choice. Precision
    // controls only a five-bit selector, not the population-count carry tree.
    Mux1H(Seq(4,8,16,32,64).map { w=>
      val count=if(w==64)Cat(0.U(4.W),bits(0))else LanePopulation.count(bits(64/w-1,0),preserveTree).pad(5)
      widthMatches(f,w)->count
    })
  }
}
class PopulationCount(n:Int) extends BlackBox {
  override def desiredName=s"SimdPopulationCount$n"
  val io=IO(new Bundle {val bits=Input(UInt(n.W));val count=Output(UInt(log2Ceil(n+1).W))})
}
object LanePopulation {
  def count(bits:UInt,preserveTree:Boolean):UInt = if(!preserveTree)PopCount(bits)else {
    require(Set(2,4,8,16).contains(bits.getWidth))
    val tree=Module(new PopulationCount(bits.getWidth));tree.io.bits:=bits;tree.io.count
  }
}
case class Config(stages:Int=2, rows:Int=8, cols:Int=8, contexts:Int=16, hop:Int=1, sramReadLatency:Int=Architecture.sramReadLatency) {
  val commitStages:Int=if(stages==2)0 else 1
  val stateReadStages:Int=if(stages==2)0 else 2
  val auxiliaryInputStages:Int=if(stages==2)0 else 2
  val peReadLatency:Int=if(stages==2)0 else 2
  val arrayReadLatency:Int=if(stages==2)0 else peReadLatency+hop*log2Ceil(math.max(rows,cols))
  val seedWriteLatency:Int=if(stages==2)1 else 2
  val busyReleaseLatency:Int=if(stages==2)0 else 1
  // A context can be accepted on the clock after its response commits. HF
  // includes two operand-read registers and one commit register explicitly.
  val feedbackInterval:Int=if(stages==2)4 else stages+commitStages+stateReadStages+1
  val peResponseCredits:Int=if(stages==2)0 else stages+commitStages+stateReadStages
  require(stages==Architecture.stages, "only two-stage F2 arithmetic is supported")
  require(contexts>=feedbackInterval && isPow2(contexts))
  require(contexts==16,"the public context ID and state bank implement exactly 16 contexts")
  require(rows>0 && rows<=8 && cols>0 && cols<=8,"the implemented buffers and address fields support up to 8x8 PEs")
  require(Set(1,2).contains(hop))
  require(sramReadLatency==Architecture.sramReadLatency,"SRAM macro read latency is one memory clock")
}
class Request extends Bundle {
  val a=UInt(64.W); val b=UInt(64.W); val acc=UInt(512.W)
  val fmt=UInt(4.W); val op=UInt(3.W); val mask=UInt(16.W)
  val ctx=UInt(4.W); val tag=UInt(32.W)
}
class ExpandedFloating extends Bundle {
  val a=UInt(512.W);val b=UInt(512.W);val c=UInt(512.W)
  val wideA=UInt(64.W);val wideB=UInt(64.W);val wideC=UInt(64.W)
}
class ExpandedSources extends Bundle { val a=UInt(512.W);val b=UInt(512.W) }
class PackedAReplicas extends BlackBox {
  override def desiredName="SimdPackedAReplicas"
  val io=IO(new Bundle {val clock=Input(Clock());val data=Input(UInt(32.W));val value=Output(UInt(512.W))})
}
class PreparedOperands extends Bundle {
  val request=new Request;val floating=new ExpandedFloating
}
class Response extends Bundle {
  val data=UInt(512.W); val flags=UInt(5.W); val overflow=UInt(16.W)
  val illegal=Bool(); val ctx=UInt(4.W); val tag=UInt(32.W)
}

/** Structure-preserving carry tree for wide HF arithmetic. */
class PrefixAdder(width:Int) extends BlackBox(Map("W"->width)) {
  override def desiredName="SimdPrefixAdd"
  val io=IO(new Bundle {
    val a=Input(UInt(width.W));val b=Input(UInt(width.W));val cin=Input(Bool())
    val result=Output(UInt((width+1).W))
  })
}
object FastCarry {
  def add(a:UInt,b:UInt,width:Int,cin:Bool=false.B):UInt = {
    val u=Module(new PrefixAdder(width))
    u.io.a:=a;u.io.b:=b;u.io.cin:=cin;u.io.result(width-1,0)
  }
  def negate(x:UInt,width:Int):UInt=add(~x,0.U,width,true.B)
}
class LeadingZeroCounter(width:Int) extends BlackBox(Map("WIDTH"->width,"MODE"->1)) {
  override def desiredName="lzc"
  val io=IO(new Bundle {
    val in_i=Input(UInt(width.W));val cnt_o=Output(UInt(math.max(1,log2Ceil(width)).W));val empty_o=Output(Bool())
  })
}
class ResultControlReplicas extends BlackBox {
  override def desiredName="SimdControlReplicas"
  val io=IO(new Bundle {
    val clock=Input(Clock());val data=Input(UInt(29.W));val value=Output(UInt(464.W))
  })
}
class ReadControlReplicas extends BlackBox {
  override def desiredName="SimdReadControlReplicas"
  val io=IO(new Bundle {
    val clock=Input(Clock());val data=Input(UInt(2.W));val value=Output(UInt(32.W))
  })
}
class BypassControlReplicas extends BlackBox {
  override def desiredName="SimdBypassControlReplicas"
  val io=IO(new Bundle {
    val clock=Input(Clock());val reset=Input(Bool());val data=Input(Bool());val value=Output(UInt(16.W))
  })
}
class OpControlRegister extends BlackBox {
  override def desiredName="SimdOpControlRegister"
  val io=IO(new Bundle {
    val clock=Input(Clock());val data=Input(UInt(3.W));val value=Output(UInt(3.W))
  })
}
class SourceFormatReplicas extends BlackBox {
  override def desiredName="SimdSourceFormatReplicas"
  val io=IO(new Bundle {val fmt=Input(UInt(4.W));val value=Output(UInt(176.W))})
}
class SourceOpReplicas extends BlackBox {
  override def desiredName="SimdSourceOpReplicas"
  val io=IO(new Bundle {val op=Input(UInt(3.W));val value=Output(UInt(128.W))})
}
class PreparationControlReplicas extends BlackBox {
  override def desiredName="SimdPreparationControlReplicas"
  val io=IO(new Bundle {
    val clock=Input(Clock());val fmt=Input(UInt(4.W));val op=Input(UInt(3.W));val value=Output(UInt(304.W))
  })
}
class ElasticPreparationControlReplicas extends BlackBox {
  override def desiredName="SimdElasticPreparationControlReplicas"
  val io=IO(new Bundle {
    val clock=Input(Clock());val enable=Input(Bool());val fmt=Input(UInt(4.W));val op=Input(UInt(3.W))
    val value=Output(UInt(304.W))
  })
}
class ResponseDataBank(depth:Int) extends BlackBox {
  override def desiredName=s"SimdResponseBank$depth"
  val io=IO(new Bundle {
    val clock=Input(Clock());val reset=Input(Bool());val write=Input(Bool());val ready=Input(Bool())
    val data=Input(UInt(32.W));val value=Output(UInt(32.W))
  })
}
class ValidReplicas extends BlackBox {
  override def desiredName="SimdValidReplicas"
  val io=IO(new Bundle {
    val clock=Input(Clock());val reset=Input(Bool());val data=Input(Bool());val value=Output(UInt(16.W))
  })
}

object Pipes {
  def delay[T <: Data](in:T, n:Int, ce:Bool):T = {
    (0 until n).foldLeft(in)((x,_)=>RegEnable(x,ce))
  }
  def compress(xs:Seq[UInt], target:Int, width:Int):Seq[UInt] = {
    var ns=xs
    while(ns.length>target) {
      ns=ns.grouped(3).flatMap {
        case Seq(a,b,c) => Seq((a^b^c)(width-1,0),(((a&b)|(a&c)|(b&c))<<1)(width-1,0))
        case rest => rest
      }.toSeq
    }
    ns
  }
  def product3(a:UInt,b:UInt,w:Int,ce:Bool):UInt = {
    val pp=(0 until w).map(i=>((a & Fill(w,b(i))).pad(2*w)<<i)(2*w-1,0))
    // Balance CSA logic depth across S2/S3: the old w/4 cut left six
    // compression levels in INT64's second stage, versus four in its first.
    val cut=Map(4->3,8->4,16->6,32->8,64->12).getOrElse(w,math.max(2,w/4))
    val s1=compress(pp,cut,2*w).map(x=>RegEnable(x,ce))
    val s2=compress(s1,2,2*w).map(x=>RegEnable(x,ce))
    RegEnable(FastCarry.add(s2(0),s2(1),2*w),ce)
  }
  def product4(a:UInt,b:UInt,w:Int,ce:Bool):UInt = {
    val pp=(0 until w).map(i=>((a & Fill(w,b(i))).pad(2*w)<<i)(2*w-1,0))
    // HF12 spends one of its former output delay registers inside the CSA
    // tree. INT64 now has 3/4/3 compressor levels, rather than 5/5.
    val cuts=Map(4->(3,2),8->(6,3),16->(8,4),32->(12,4),64->(22,6))(w)
    val s1=compress(pp,cuts._1,2*w).map(x=>RegEnable(x,ce))
    val s2=compress(s1,cuts._2,2*w).map(x=>RegEnable(x,ce))
    val s3=compress(s2,2,2*w).map(x=>RegEnable(x,ce))
    RegEnable(FastCarry.add(s3(0),s3(1),2*w),ce)
  }
}

/** Exact source expansion. The FMA performs the only result rounding. */
object ExpandFloat {
  def ieee(x:UInt,e:Int,m:Int):UInt = {
    val sign=x(e+m); val exp=x(e+m-1,m); val frac=x(m-1,0)
    // Source expansion was an extracted input critical path: priority
    // encode -> increment/subtract -> variable shift. Each subnormal's
    // leading position instead selects constant exponent and fixed wiring.
    // The mutually exclusive terms preserve every source bit, including
    // signaling-NaN payloads; no rounding occurs here.
    val base=127-((1<<(e-1))-1)
    val subnormal=Mux1H((0 until m).map {position=>
      val leading=frac(position) && (if(position==m-1)true.B else !frac(m-1,position+1).orR)
      val mantissa=if(position==0)0.U(23.W) else Cat(frac(position-1,0),0.U((23-position).W))
      leading->Cat(sign,(base-(m-1-position)).U(8.W),mantissa)
    })
    val normalExp=if(e==5)Cat(Mux(exp(4),8.U(4.W),7.U(4.W)),exp(3,0))
      else (exp.pad(9)+base.U)(7,0)
    Mux(exp.andR,Cat(sign,255.U(8.W),frac,0.U((23-m).W)),
      Mux(exp===0.U,Mux(frac===0.U,Cat(sign,0.U(31.W)),subnormal),
        Cat(sign,normalExp(7,0),frac,0.U((23-m).W))))
  }
  def apply(x:UInt,fmt:UInt):UInt = {
    val f4=VecInit(Seq(0x00000000L,0x3f000000L,0x3f800000L,0x3fc00000L,
      0x40000000L,0x40400000L,0x40800000L,0x40c00000L).map(_.U(32.W)))
    MuxLookup(fmt,x(31,0))(Seq(
      5.U->(f4(x(2,0))|Cat(x(3),0.U(31.W))),
      6.U->ieee(x(7,0),5,2), 7.U->ieee(x(15,0),5,10),
      8.U->Cat(x(15,0),0.U(16.W))))
  }
}

/** RNE arithmetic preparation before the registered PE operand boundary.
  * MUL uses the CVFPU RNE negative-zero addend; ADD/SUB use exact +1.
  * Masked channels retain their original accumulator representation. */
object PrepareFloating {
  def apply(in:Request):Request = {
    val out=WireDefault(in)
    val ones=Seq("2","3c","3c00","3f80","3f800000","3ff0000000000000").map(BigInt(_,16))
    when(in.fmt>=5.U && in.fmt<=10.U && in.op<=3.U && in.op=/=1.U) {
      out.op:=1.U
      for(f<-5 to 10)when(in.fmt===f.U) {
        val w=Formats.widths(f);val dw=if(f==10)64 else 32
        when(in.op>=2.U){out.a:=Fill(64/w,ones(f-5).U(w.W));out.b:=in.a}
        val values=(0 until 64/w).map {lane=>
          val b=if(f==10)in.b else ExpandFloat(in.b((lane+1)*w-1,lane*w).pad(32),f.U)
          val addend=b ^ Mux(in.op===3.U,(BigInt(1)<<(dw-1)).U(dw.W),0.U(dw.W))
          val value=Mux(in.op===0.U,(BigInt(1)<<(dw-1)).U(dw.W),addend)
          Mux(in.mask(lane),value,in.acc((lane+1)*dw-1,lane*dw))
        }
        out.acc:=Cat(values.reverse).pad(512)
      }
    }
    out
  }
}

/** Expand source encodings before the existing operand register. IEEE input
  * classification then starts from registered FP32/FP64 values, rather than
  * following the subnormal decoder and runtime format selection. */
object ExpandSources {
  def apply(in:Request,broadcastA:Boolean=false,control:Option[UInt]=None,packedA:Option[Seq[UInt]]=None):ExpandedSources = {
    require(packedA.isEmpty || (broadcastA && packedA.get.length==16))
    val out=Wire(new ExpandedSources)
    val formats=if(control.isDefined)None else Some(Module(new SourceFormatReplicas))
    formats.foreach(_.io.fmt:=in.fmt)
    def isFmt(lane:Int,f:Int):Bool=control.map(_(19*lane+f)).getOrElse(formats.get.io.value(11*lane+f))
    val lanes=(0 until 16).map {lane=>
      val supported=(5 to 9).filter(f=>lane<64/Formats.widths(f))
      def operand(value:UInt,broadcast:Boolean):UInt={
        val terms=supported.map {f=>
          val w=Formats.widths(f)
          val element=if(broadcast)value(w-1,0)else value((lane+1)*w-1,lane*w)
          (isFmt(lane,f)&&in.mask(lane))->ExpandFloat(element.pad(32),f.U)
        }
        // Mux1H's single-input case forwards its data unconditionally.
        // Lanes 8..15 support only FP4, so explicitly preserve isolation
        // when another format or a masked FP4 lane is selected.
        if(terms.size==1)Mux(terms.head._1,terms.head._2,0.U(32.W))else Mux1H(terms)
      }
      (operand(packedA.map(_(lane)).getOrElse(in.a),broadcastA),operand(in.b,false))
    }
    out.a:=Cat(lanes.reverse.map(_._1));out.b:=Cat(lanes.reverse.map(_._2));out
  }
}
object PrepareOperands {
  def apply(in:Request,broadcastA:Boolean=false,source:Option[ExpandedSources]=None,control:Option[UInt]=None):PreparedOperands = {
    val out=Wire(new PreparedOperands);val q=in
    out.request:=q
    val expanded=source.getOrElse(ExpandSources(q,broadcastA))
    val formats=if(control.isDefined)None else Some(Module(new SourceFormatReplicas))
    formats.foreach(_.io.fmt:=q.fmt)
    val operations=if(broadcastA||control.isDefined)None else Some(Module(new SourceOpReplicas))
    operations.foreach(_.io.op:=q.op)
    def isFmt(lane:Int,f:Int):Bool=control.map(_(19*lane+f)).getOrElse(formats.get.io.value(11*lane+f))
    def isOp(lane:Int,op:Int):Bool=if(broadcastA)false.B
      else control.map(_(19*lane+11+op)).getOrElse(operations.get.io.value(8*lane+op))
    def normalize(lane:Int,a:UInt,b:UInt,c:UInt,one:BigInt,sign:BigInt):(UInt,UInt,UInt) = {
      val mul=isOp(lane,0);val add=isOp(lane,2)
      val sub=isOp(lane,3);val addOrSub=add||sub
      (
        Mux(addOrSub,one.U(32.W),a),Mux(addOrSub,a,b),
        Mux(mul,sign.U(32.W),Mux(addOrSub,b^Mux(sub,sign.U(32.W),0.U(32.W)),c)))
    }
    val lanes=(0 until 16).map {lane=>
      val enabled=(5 to 9).filter(f=>lane<64/Formats.widths(f)).map(f=>isFmt(lane,f)).reduce(_||_) && q.mask(lane)
      val a=expanded.a(32*lane+31,32*lane);val b=expanded.b(32*lane+31,32*lane)
      val c=Mux(enabled,q.acc(32*lane+31,32*lane),0.U(32.W))
      val normal=normalize(lane,a,b,c,BigInt("3f800000",16),BigInt("80000000",16))
      // ExpandSources already isolates both source operands by format and
      // lane mask. A MAC-only PE needs no second independently decoded gate
      // after expansion; only the accumulator still needs isolation here.
      (if(broadcastA)a else Mux(enabled,normal._1,0.U(32.W)),
       if(broadcastA)b else Mux(enabled,normal._2,0.U(32.W)),Mux(enabled,normal._3,0.U(32.W)))
    }
    out.floating.a:=Cat(lanes.reverse.map(_._1))
    out.floating.b:=Cat(lanes.reverse.map(_._2))
    out.floating.c:=Cat(lanes.reverse.map(_._3))
    // FP64 has its own registered operands. Sharing the first two FP32 words
    // allowed inactive units to see live data and merged their input FFs.
    val wide=(0 until 2).map {word=>
      val enabled=isFmt(word,10) && q.mask(0)
      val normal=normalize(word,q.a(32*word+31,32*word),q.b(32*word+31,32*word),q.acc(32*word+31,32*word),
        if(word==0)BigInt(0)else BigInt("3ff00000",16),if(word==0)BigInt(0)else BigInt("80000000",16))
      (Mux(enabled,normal._1,0.U(32.W)),Mux(enabled,normal._2,0.U(32.W)),Mux(enabled,normal._3,0.U(32.W)))
    }
    out.floating.wideA:=Cat(wide.reverse.map(_._1));out.floating.wideB:=Cat(wide.reverse.map(_._2))
    out.floating.wideC:=Cat(wide.reverse.map(_._3))
    out
  }
}

class ScalarFp(w:Int,stages:Int) extends BlackBox {
  override def desiredName=s"SimdFp${if(stages==2) "F2" else if(stages==8) "HF" else "HF12_"}$w"
  val io=IO(new Bundle {
    val clock=Input(Clock()); val ce=Input(Bool())
    val a=Input(UInt(w.W));val b=Input(UInt(w.W));val c=Input(UInt(w.W))
    val op=Input(UInt(3.W));val result=Output(UInt(w.W));val flags=Output(UInt(5.W))
  })
}

object IntegerToFloat {
  def apply(x:UInt,w:Int,stages:Int,ce:Bool):(UInt,UInt) = {
    val sign=x(w-1);val neg=if(stages==2)(-x.asSInt).asUInt else FastCarry.negate(x,w)
    val mag=Mux(sign,neg,x)
    val s1=RegEnable(sign,ce);val m1=RegEnable(mag,ce)
    val leading=if(stages==2)PriorityEncoder(Reverse(mag)) else {
      val tree=Module(new LeadingZeroCounter(w));tree.io.in_i:=m1;tree.io.cnt_o
    }
    val idx=RegEnable((w-1).U-leading,ce)
    val m=if(stages==2)m1 else RegEnable(m1,ce)
    val s=if(stages==2)s1 else RegEnable(s1,ce)
    val right=if(stages==2)Mux(idx>23.U,idx-23.U,0.U)else
      RegEnable(Mux(leading<(w-24).U,(w-24).U-leading,0.U),ce)
    val left=if(stages==2)Mux(idx<23.U,23.U-idx,0.U)else
      RegEnable(Mux(leading>(w-24).U,leading-(w-24).U,0.U),ce)
    val guardIndex=if(stages==2)right-1.U else RegEnable((w-25).U-leading,ce)
    val keep=Mux(idx>23.U,(m>>right)(23,0),(m.pad(math.max(w,24))<<left)(23,0))
    val guard=right=/=0.U && (m>>guardIndex)(0)
    val sticky=if(stages==2) {
      val lowMask=((1.U((w+1).W)<<Mux(right===0.U,0.U,right-1.U))-1.U)(w-1,0)
      right>1.U && (m&lowMask).orR
    }else (0 until w).map(i=>m(i) && right>(i+1).U).reduce(_||_)
    val exp=idx+&127.U
    val zero= !m.orR
    val (k,g,t,e,z,sgn)=if(stages==2)(keep,guard,sticky,exp,zero,s) else
      (RegEnable(keep,ce),RegEnable(guard,ce),RegEnable(sticky,ce),RegEnable(exp,ce),RegEnable(zero,ce),RegEnable(s,ce))
    val increment=g&&(t||k(0))
    val rounded=if(stages==2)k+&increment else FastCarry.add(k.pad(25),0.U,25,increment)
    val exponent=e+rounded(24)
    val overflow=exponent>=255.U
    val data=Mux(z,0.U(32.W),Mux(overflow,Cat(sgn,255.U(8.W),0.U(23.W)),
      Cat(sgn,exponent(7,0),Mux(rounded(24),rounded(23,1),rounded(22,0)))))
    val flags=Mux(z,0.U(5.W),Cat(0.U(2.W),overflow,false.B,g||t||overflow))
    val result=RegEnable(data,ce);val status=RegEnable(flags,ce)
    if(stages==2)(result,status) else (Pipes.delay(result,stages-4,ce),Pipes.delay(status,stages-4,ce))
  }
}

/** F2 uses an elastic pipeline; HF reserves response credits before issuing. */
class SimdAlu(cfg:Config,streaming:Boolean=false,predecodedFloat:Boolean=false,reservedOutput:Boolean=false) extends Module with RequireSyncReset {
  val io=IO(new Bundle {
    val in=Flipped(Decoupled(new Request));val out=Decoupled(new Response)
    val floating=if(predecodedFloat)Some(Input(new ExpandedFloating))else None
    val wordValid=if(reservedOutput)Some(Output(UInt(16.W)))else None
  })
  val valid=RegInit(VecInit(Seq.fill(cfg.stages)(false.B)))
  val computed=Wire(Decoupled(new Response))
  val ce=if(cfg.stages==2) {
    io.out<>computed
    val advance= !valid.last || computed.ready
    // Memory runs at half the core rate. Once all valid work has drained,
    // hold the arithmetic until another request arrives; admission and the
    // two-cycle latency are unchanged. Do not clock dummy FMAs during DMA.
    io.in.ready:=advance;advance && (io.in.valid || valid.asUInt.orR)
  }else if(streaming || reservedOutput) {
    io.out<>computed;io.in.ready:=true.B
    when(computed.valid){assert(computed.ready,"streaming arithmetic sink must accept every result")}
    true.B
  }else {
    // Reserve a response slot at issue time. Once launched, an HF arithmetic
    // operation advances every clock; external ready never fans out through
    // all arithmetic registers. A flow queue preserves unblocked latency.
    val capacity=cfg.stages+2
    val stored=RegInit(0.U(log2Ceil(capacity+1).W));val empty=RegInit(true.B)
    val enqueue=computed.valid && !(empty && io.out.ready)
    val dequeue= !empty && io.out.ready
    val emptyNext=Mux(enqueue=/=dequeue,Mux(enqueue,false.B,stored===1.U),empty)
    empty:=emptyNext
    when(enqueue && !dequeue){stored:=stored+1.U}
    when(!enqueue && dequeue){stored:=stored-1.U}
    val responseWidth=(new Response).getWidth
    val inputWords=computed.bits.asUInt.pad(((responseWidth+31)/32)*32)
    val words=(0 until (responseWidth+31)/32).map {word=>
      val bank=Module(new ResponseDataBank(capacity));bank.io.clock:=clock;bank.io.reset:=reset.asBool
      bank.io.write:=enqueue;bank.io.ready:=io.out.ready;bank.io.data:=inputWords(32*word+31,32*word)
      bank.io.value
    }
    val buffered=Cat(words.reverse)(responseWidth-1,0).asTypeOf(new Response)
    io.out.valid:= !empty || computed.valid
    io.out.bits:=Mux(empty,computed.bits,buffered)
    // Register the empty decision beside each result word. A pointer equality
    // tree must not drive all 512 data bypass muxes before the commit register.
    val bypass=Module(new BypassControlReplicas)
    bypass.io.clock:=clock;bypass.io.reset:=reset.asBool;bypass.io.data:=emptyNext
    io.out.bits.data:=Cat((0 until 16).reverse.map {word=>
      Mux(bypass.io.value(word),computed.bits.data(32*word+31,32*word),buffered.data(32*word+31,32*word))
    })
    computed.ready:=stored<capacity.U || dequeue
    assert(empty===(stored===0.U) && stored<=capacity.U)
    when(enqueue){assert(computed.ready,"reserved response buffer overflow")}
    val credits=RegInit(capacity.U(log2Ceil(capacity+1).W))
    io.in.ready:=credits=/=0.U
    when(io.in.fire && !io.out.fire){credits:=credits-1.U}
    when(!io.in.fire && io.out.fire){credits:=credits+1.U}
    assert(credits<=capacity.U)
    when(computed.valid){assert(computed.ready,"reserved HF response slot was lost")}
    true.B
  }
  when(ce) { valid(0):=io.in.fire; for(i<-1 until cfg.stages) valid(i):=valid(i-1) }
  computed.valid:=valid.last
  if(reservedOutput) {
    require(cfg.stages>2)
    val copies=Module(new ValidReplicas)
    copies.io.clock:=clock;copies.io.reset:=reset.asBool;copies.io.data:=valid(cfg.stages-2)
    io.wordValid.get:=copies.io.value
    for(word<-0 until 16)assert(copies.io.value(word)===computed.valid,"local arithmetic validity diverged")
  }
  if(streaming)when(io.in.fire){assert(io.in.bits.op===1.U,"array streaming arithmetic implements MAC")}
  val q=Pipes.delay(io.in.bits,cfg.stages,ce)
  // Decode alongside the arithmetic, rather than decoding the shared late
  // format register through every result and event-counter selection path.
  val formatOH=Pipes.delay(UIntToOH(io.in.bits.fmt,11),cfg.stages,ce)
  def isFmt(f:Int):Bool=if(cfg.stages==2)q.fmt===f.U else formatOH(f)
  val integer=Pipes.delay(io.in.bits.fmt<5.U,cfg.stages,ce)
  val convert=Pipes.delay(io.in.bits.op===7.U,cfg.stages,ce)
  val illegal=Pipes.delay(io.in.bits.fmt>10.U || (io.in.bits.fmt>=5.U && io.in.bits.op>3.U),cfg.stages,ce)
  val flagMask=Pipes.delay(io.in.bits.mask & Formats.mask(io.in.bits.fmt),cfg.stages,ce)
  val intData=WireDefault(0.U(512.W));val intOverflow=WireDefault(0.U(16.W))
  val convertData=WireDefault(0.U(512.W));val convertFlags=WireDefault(0.U(5.W))
  val integerChoices=Wire(Vec(5,UInt(512.W)));val conversionChoices=Wire(Vec(5,UInt(512.W)))

  for((w,f)<-Formats.widths.take(5).zipWithIndex) {
    val aw=Formats.accs(f)
    val conversions=if(streaming)Seq.fill(64/w)((0.U(32.W),0.U(5.W)))else (0 until 64/w).map {lane=>
      val (data,flags)=IntegerToFloat(io.in.bits.acc((lane+1)*aw-1,lane*aw),aw,cfg.stages,ce)
      (if(cfg.stages==2)Mux(q.mask(lane),data,0.U(32.W))else data,Mux(q.mask(lane),flags,0.U(5.W)))
    }
    conversionChoices(f):=Cat(conversions.reverse.map(_._1)).pad(512)
    when(isFmt(f)) {
      convertData:=Cat(conversions.reverse.map(_._1))
      convertFlags:=conversions.map(_._2).reduce(_|_)
    }
    val pieces=(0 until 64/w).map { lane=>
      val active=io.in.bits.fmt===f.U && io.in.bits.mask(lane)
      val a=Mux(active,if(streaming)io.in.bits.a(w-1,0)else io.in.bits.a((lane+1)*w-1,lane*w),0.U).asSInt
      val b=Mux(active,io.in.bits.b((lane+1)*w-1,lane*w),0.U).asSInt
      val acc=io.in.bits.acc((lane+1)*aw-1,lane*aw).asSInt
      val op=io.in.bits.op
      val (prod,old,aa,bb,oo)=if(cfg.stages==2) {
        (RegEnable(a*b,ce),RegEnable(acc,ce),RegEnable(a,ce),RegEnable(b,ce),RegEnable(op,ce))
      } else {
        val a1=RegEnable(Mux(a<0.S,FastCarry.negate(a.asUInt,w),a.asUInt),ce)
        val b1=RegEnable(Mux(b<0.S,FastCarry.negate(b.asUInt,w),b.asUInt),ce)
        val productLatency=if(cfg.stages==12)5 else 4
        val sign=Pipes.delay(a(w-1)^b(w-1),productLatency,ce)
        val magnitude=if(cfg.stages==12)Pipes.product4(a1,b1,w,ce)else Pipes.product3(a1,b1,w,ce)
        val signedProduct=RegEnable(Mux(sign,FastCarry.negate(magnitude,2*w).asSInt,magnitude.asSInt),ce)
        val localOp=if(streaming)1.U(3.W)else {
          val reg=Module(new OpControlRegister);reg.io.clock:=clock
          reg.io.data:=Pipes.delay(op,productLatency,ce);reg.io.value
        }
        (signedProduct,Pipes.delay(acc,productLatency+1,ce),Pipes.delay(a,productLatency+1,ce),Pipes.delay(b,productLatency+1,ce),localOp)
      }
      val fastSum=if(cfg.stages==2)0.S else {
        val first=Mux(oo===1.U,prod.pad(aw+1),aa.pad(aw+1)).asUInt
        val second=Mux(oo===1.U,old.pad(aw+1),bb.pad(aw+1)).asUInt
        val subtract=oo===3.U
        // HF8 S6/S7 and HF12 S7/S8 separate operand selection from carry.
        val left=RegEnable(first,ce);val right=RegEnable(Mux(subtract,~second,second),ce)
        val carry=RegEnable(subtract,ce)
        FastCarry.add(left,right,aw+1,carry).asSInt
      }
      val selectedOp=if(cfg.stages==2)oo else RegEnable(oo,ce)
      val multiply=if(cfg.stages==2)prod else RegEnable(prod,ce)
      val bitAnd=if(cfg.stages==2)aa.asUInt&bb.asUInt else RegEnable(aa.asUInt&bb.asUInt,ce)
      val bitOr=if(cfg.stages==2)aa.asUInt|bb.asUInt else RegEnable(aa.asUInt|bb.asUInt,ce)
      val bitXor=if(cfg.stages==2)aa.asUInt^bb.asUInt else RegEnable(aa.asUInt^bb.asUInt,ce)
      val full=Wire(SInt((aw+1).W));full:=0.S
      switch(selectedOp) {
        is(0.U){full:=multiply.pad(aw+1)}
        is(1.U){full:=(if(cfg.stages==2)prod.pad(aw+1)+old.pad(aw+1)else fastSum)}
        is(2.U){full:=(if(cfg.stages==2)aa.pad(aw+1)+bb.pad(aw+1)else fastSum)}
        is(3.U){full:=(if(cfg.stages==2)aa.pad(aw+1)-bb.pad(aw+1)else fastSum)}
        is(4.U){full:=bitAnd.zext}
        is(5.U){full:=bitOr.zext}
        is(6.U){full:=bitXor.zext}
      }
      val reg=RegEnable(full.asUInt,ce)
      val result=if(cfg.stages==2) reg else Pipes.delay(reg,cfg.stages-(if(cfg.stages==12)8 else 7),ce)
      val masked=if(cfg.stages==2)Mux(q.mask(lane),result(aw-1,0),q.acc((lane+1)*aw-1,lane*aw))else result(aw-1,0)
      (masked,(result(aw)^result(aw-1)) && q.mask(lane))
    }
    integerChoices(f):=Cat(pieces.reverse.map(_._1)).pad(512)
    when(isFmt(f)) {
      intData:=Cat(pieces.reverse.map(_._1)).pad(512)
      intOverflow:=Cat(pieces.reverse.map(_._2))
    }
  }
  val fps=(0 until 16).map { lane=>
    val a=WireDefault(0.U(32.W));val b=WireDefault(0.U(32.W))
    for(f<-5 to 9 if lane<64/Formats.widths(f)) {
      val w=Formats.widths(f)
      when(io.in.bits.fmt===f.U && io.in.bits.mask(lane)) {
        a:=ExpandFloat((if(streaming)io.in.bits.a(w-1,0)else io.in.bits.a((lane+1)*w-1,lane*w)).pad(32),f.U)
        b:=ExpandFloat(io.in.bits.b((lane+1)*w-1,lane*w).pad(32),f.U)
      }
    }
    val u=Module(new ScalarFp(32,cfg.stages))
    u.io.clock:=clock;u.io.ce:=ce
    u.io.a:=(if(predecodedFloat)io.floating.get.a(32*lane+31,32*lane)else a)
    u.io.b:=(if(predecodedFloat)io.floating.get.b(32*lane+31,32*lane)else b)
    u.io.c:=(if(predecodedFloat)io.floating.get.c(32*lane+31,32*lane)else
      Mux(io.in.bits.fmt>=5.U && io.in.bits.fmt<=9.U && Formats.mask(io.in.bits.fmt)(lane) && io.in.bits.mask(lane),io.in.bits.acc(32*lane+31,32*lane),0.U))
    u.io.op:=(if(predecodedFloat)1.U else io.in.bits.op)
    val selected=io.in.bits.fmt>=5.U && io.in.bits.fmt<=9.U && Formats.mask(io.in.bits.fmt)(lane) && io.in.bits.mask(lane)
    when(io.in.fire && !selected){assert(u.io.a===0.U && u.io.b===0.U && u.io.c===0.U,"inactive FP32 lane saw live operands")}
    (if(cfg.stages==2)Mux(q.mask(lane),u.io.result,q.acc(32*lane+31,32*lane))else u.io.result,u.io.flags)
  }
  val d=Module(new ScalarFp(64,cfg.stages))
  val active64=io.in.bits.fmt===10.U && io.in.bits.mask(0)
  d.io.clock:=clock;d.io.ce:=ce;d.io.op:=(if(predecodedFloat)1.U else io.in.bits.op)
  d.io.a:=(if(predecodedFloat)io.floating.get.wideA else Mux(active64,io.in.bits.a,0.U))
  d.io.b:=(if(predecodedFloat)io.floating.get.wideB else Mux(active64,io.in.bits.b,0.U))
  d.io.c:=(if(predecodedFloat)io.floating.get.wideC else Mux(active64,io.in.bits.acc(63,0),0.U))
  when(io.in.fire && !active64){assert(d.io.a===0.U && d.io.b===0.U && d.io.c===0.U,"inactive FP64 unit saw live operands")}
  val laneFlags=fps.zipWithIndex.map {case((_,flags),i)=>
    Mux(flagMask(i),flags,0.U)
  }.reduce(_|_)
  val fpData=Cat(fps.reverse.map(_._1))
  val trimmed=WireDefault(fpData)
  for(f<-5 to 9) {
    val bits=64/Formats.widths(f)*32
    when(isFmt(f)) { trimmed:=fpData(bits-1,0).pad(512) }
  }
  if(cfg.stages==2)computed.bits.data:=Mux(integer,Mux(convert,convertData,intData),
    Mux(isFmt(10),Mux(q.mask(0),d.io.result,q.acc(63,0)).pad(512),trimmed))
  else {
    val controls=Module(new ResultControlReplicas);controls.io.clock:=clock
    controls.io.data:=Pipes.delay(Cat(io.in.bits.mask,UIntToOH(io.in.bits.fmt,11),io.in.bits.fmt<5.U,io.in.bits.op===7.U),cfg.stages-1,ce)
    val words=(0 until 16).map {word=>
      val local=controls.io.value(29*word+28,29*word);val formats=local(12,2);val mask=local(28,13)
      // Each slice uses its own mask FFs. Wide integer lanes may straddle
      // a 32-bit boundary (48/80/144-bit accumulators), so select per bit
      // using the format's static lane index and zero unused high bits.
      val intWord=Mux1H((0 until 5).map {f=>
        val aw=Formats.accs(f);val used=64/Formats.widths(f)*aw
        val data=integerChoices(f)(32*word+31,32*word)
        val masked=Cat((0 until 32).reverse.map {bit=>
          val position=word*32+bit
          if(position>=used)false.B else Mux(mask(position/aw),data(bit),q.acc(position))
        })
        formats(f)->masked
      })
      val converted=if(streaming)0.U(32.W)else Mux(mask(word),Mux1H((0 until 5).map(f=>formats(f)->conversionChoices(f)(32*word+31,32*word))),0.U(32.W))
      val fpActive=(5 to 9).filter(f=>word<64/Formats.widths(f)).map(formats(_)).reduceOption(_||_).getOrElse(false.B)
      val fpWord=Mux(fpActive,Mux(mask(word),fpData(32*word+31,32*word),q.acc(32*word+31,32*word)),0.U(32.W))
      val wide=if(word<2)Mux(mask(0),d.io.result(32*word+31,32*word),q.acc(32*word+31,32*word))else 0.U(32.W)
      Mux(local(1),if(streaming)intWord else Mux(local(0),converted,intWord),Mux(formats(10),wide,fpWord))
    }
    computed.bits.data:=Cat(words.reverse)
  }
  computed.bits.flags:=Mux(integer,if(streaming)0.U else Mux(convert,convertFlags,0.U),Mux(isFmt(10),Mux(q.mask(0),d.io.flags,0.U),laneFlags))
  computed.bits.overflow:=Mux(integer && !convert,intOverflow,0.U)
  computed.bits.illegal:=illegal
  computed.bits.ctx:=q.ctx;computed.bits.tag:=q.tag
}
