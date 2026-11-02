"""An isolated integer conversion candidate at the existing two F2 cuts."""
SCALA='''object IntegerToFloat {
  def apply(x:UInt,w:Int,stages:Int,ce:Bool):(UInt,UInt) = {
    require(stages==2)
    val sign=x(w-1)
    val mag=Mux(sign,FastCarry.negate(x,w),x)
    val position=(w-1).U-PriorityEncoder(Reverse(mag))
    // Prepare binary shift controls and the exponent before the existing cut.
    val s=RegEnable(sign,ce);val m=RegEnable(mag,ce)
    val useRight=RegEnable(position>=24.U,ce)
    val right=RegEnable(Mux(position>=24.U,position-24.U,0.U),ce)
    val left=RegEnable(Mux(position<23.U,23.U-position,0.U),ce)
    val exponent=RegEnable(position+&127.U,ce)
    val hasSticky=RegEnable(position>=25.U,ce)
    val stickyIndex=RegEnable(Mux(position>=25.U,position-25.U,0.U),ce)
    // Lower prefixes share the magnitude register boundary. No third stage.
    var lower=(0 until w).map(i=>mag(i))
    for(level<-0 until log2Ceil(w)) {
      val previous=lower;val distance=1<<level
      lower=(0 until w).map(i=>if(i>=distance)previous(i)||previous(i-distance)else previous(i))
    }
    val low=RegEnable(Cat(lower.reverse),ce)
    val shifted=(m.pad(25)>>right)(24,0)
    val keep=Mux(useRight,shifted(24,1),(m<<left)(23,0))
    val guard=useRight && shifted(0)
    val sticky=hasSticky && (low>>stickyIndex)(0)
    val rounded=FastCarry.add(keep.pad(25),0.U(25.W),25,guard&&(sticky||keep(0)))
    val exp=FastCarry.add(exponent.pad(9),Cat(0.U(8.W),rounded(24)),9)
    val overflow=exp>=255.U;val zero= !low(w-1)
    val data=Mux(zero,0.U(32.W),Mux(overflow,Cat(s,255.U(8.W),0.U(23.W)),
      Cat(s,exp(7,0),Mux(rounded(24),rounded(23,1),rounded(22,0)))))
    val flags=Mux(zero,0.U(5.W),Cat(0.U(2.W),overflow,false.B,guard||sticky||overflow))
    (RegEnable(data,ce),RegEnable(flags,ce))
  }
}
'''


def transform(text):
    begin=text.index('object IntegerToFloat {')
    end=text.index('/** F2 uses an elastic pipeline;',begin)
    return text[:begin]+SCALA+'\n'+text[end:]
