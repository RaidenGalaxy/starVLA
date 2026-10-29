"""Independent integer/rational oracle. No host floating FMA or double rounding."""
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
import struct

class Format(IntEnum):
    I4=0; I8=1; I16=2; I32=3; I64=4
    F4=5; F8=6; F16=7; BF16=8; F32=9; F64=10

class Op(IntEnum):
    MUL=0; MAC=1; ADD=2; SUB=3; AND=4; OR=5; XOR=6; I2F=7

WIDTHS = (4,8,16,32,64,4,8,16,16,32,64)
ACC_WIDTHS = (32,32,48,80,144,32,32,32,32,32,64)
ENCODINGS = {Format.F4:(2,1), Format.F8:(5,2), Format.F16:(5,10),
             Format.BF16:(8,7), Format.F32:(8,23), Format.F64:(11,52)}
NV, DZ, OF, UF, NX = 16,8,4,2,1

def signed(x, width):
    x &= (1<<width)-1
    return x-(1<<width) if x>>(width-1) else x

@dataclass(frozen=True)
class FloatValue:
    kind: str
    sign: int
    magnitude: Fraction = Fraction(0)
    signaling: bool = False

def pow2(n):
    return Fraction(1<<n) if n>=0 else Fraction(1,1<<-n)

def decode(bits, fmt):
    fmt=Format(fmt); e,m=ENCODINGS[fmt]
    sign=bits>>(e+m)&1; exp=bits>>m&((1<<e)-1); mant=bits&((1<<m)-1)
    bias=(1<<(e-1))-1
    if fmt!=Format.F4 and exp==(1<<e)-1:
        return FloatValue('nan' if mant else 'inf',sign,signaling=bool(mant and not mant>>(m-1)))
    value=Fraction(mant,1<<m)*pow2(1-bias) if exp==0 else (1+Fraction(mant,1<<m))*pow2(exp-bias)
    return FloatValue('finite',sign,value)

def canonical_nan(fmt):
    e,m=ENCODINGS[Format(fmt)]
    return (((1<<e)-1)<<m)|(1<<(m-1))

def round_even(x):
    q,r=divmod(x.numerator,x.denominator)
    return q+int(2*r>x.denominator or (2*r==x.denominator and q&1)), bool(r)

def encode(value, fmt, zero_sign=0):
    fmt=Format(fmt); e,m=ENCODINGS[fmt]; w=1+e+m
    sign=int(value<0) if value else zero_sign
    x=abs(Fraction(value)); signbit=sign<<(w-1)
    if not x: return signbit,0
    if fmt==Format.F4:
        candidates=[decode(i,fmt).magnitude for i in range(8)]
        ix=min(range(8),key=lambda i:(abs(candidates[i]-x),i&1))
        return signbit|ix, NX if candidates[ix]!=x else 0
    bias=(1<<(e-1))-1; emin=1-bias; emax=bias
    exponent=x.numerator.bit_length()-x.denominator.bit_length()
    if x<pow2(exponent): exponent-=1
    step=pow2(max(exponent,emin)-m)
    significand,inexact=round_even(x/step)
    if exponent<emin:
        result=signbit|significand
        # IEEE tininess-after-rounding uses precision rounding with an
        # unbounded exponent, before the subnormal exponent is clamped. A
        # final minimum-normal encoding can therefore still raise UF.
        unbounded_step=pow2(exponent-m)
        unbounded_sig,_=round_even(x/unbounded_step)
        tiny=unbounded_sig*unbounded_step<pow2(emin)
        return result,(NX|UF if tiny else NX) if inexact else 0
    if significand==(1<<(m+1)):
        significand>>=1; exponent+=1
    if exponent>emax:
        return signbit|(((1<<e)-1)<<m), OF|NX
    return signbit|((exponent+bias)<<m)|(significand-(1<<m)), NX if inexact else 0

def fp_operation(a,b,c,fmt,op):
    fmt=Format(fmt); op=Op(op); dst=Format.F64 if fmt==Format.F64 else Format.F32
    va,vb,vc=decode(a,fmt),decode(b,fmt),decode(c,dst)
    if op in (Op.ADD,Op.SUB):
        va=FloatValue('finite',0,Fraction(1)); vc=decode(b,fmt)
        vb=decode(a,fmt)
        if op==Op.SUB: vc=FloatValue(vc.kind,vc.sign^1,vc.magnitude,vc.signaling)
    if op==Op.MUL:
        vc=FloatValue('finite',va.sign^vb.sign,Fraction(0))
    badprod=(va.kind=='inf' and vb.kind=='finite' and vb.magnitude==0) or (vb.kind=='inf' and va.kind=='finite' and va.magnitude==0)
    if badprod or any(v.kind=='nan' for v in (va,vb,vc)):
        return canonical_nan(dst), NV if badprod or any(v.signaling for v in (va,vb,vc)) else 0
    psign=va.sign^vb.sign
    if va.kind=='inf' or vb.kind=='inf':
        if vc.kind=='inf' and vc.sign!=psign: return canonical_nan(dst),NV
        e,m=ENCODINGS[dst]; return (psign<<(e+m))|(((1<<e)-1)<<m),0
    if vc.kind=='inf':
        e,m=ENCODINGS[dst]; return (vc.sign<<(e+m))|(((1<<e)-1)<<m),0
    product=va.magnitude*vb.magnitude*(-1 if psign else 1)
    addend=vc.magnitude*(-1 if vc.sign else 1)
    zero_sign=psign if product==0 and addend==0 and psign==vc.sign else 0
    return encode(product+addend,dst,zero_sign)

def operate(a,b,acc,fmt,op=Op.MAC,mask=0xffff):
    fmt=Format(fmt); op=Op(op); w=WIDTHS[fmt]; aw=ACC_WIDTHS[fmt]
    output=0; flags=0; overflow=0
    if op==Op.I2F:
        if fmt>=Format.F4:raise ValueError('I2F requires an integer source')
        for lane in range(64//w):
            if mask>>lane&1:
                value=signed(acc>>(lane*aw),aw)
                result,fl=encode(Fraction(value),Format.F32);flags|=fl;output|=result<<(lane*32)
        return output,flags,0
    for lane in range(64//w):
        av=a>>(lane*w)&((1<<w)-1); bv=b>>(lane*w)&((1<<w)-1)
        cv=acc>>(lane*aw)&((1<<aw)-1)
        if not mask>>lane&1: result=cv
        elif fmt>=Format.F4:
            if op>Op.SUB: raise ValueError('logical operations require integer format')
            result,fl=fp_operation(av,bv,cv,fmt,op); flags|=fl
        else:
            ai,bi,ci=signed(av,w),signed(bv,w),signed(cv,aw)
            if op==Op.MUL: result=ai*bi
            elif op==Op.MAC: result=ai*bi+ci
            elif op==Op.ADD: result=ai+bi
            elif op==Op.SUB: result=ai-bi
            elif op==Op.AND: result=av&bv
            elif op==Op.OR: result=av|bv
            else: result=av^bv
            overflow|=int(not -(1<<(aw-1))<=result<(1<<(aw-1)))<<lane
        output|=(result&((1<<aw)-1))<<(lane*aw)
    return output,flags,overflow

def pack(values, width):
    return sum((int(x)&((1<<width)-1))<<(i*width) for i,x in enumerate(values))

def to_bits(value,fmt):
    return encode(Fraction(value),Format(fmt))[0]

def narrow32(bits,dst):
    dst=Format(dst)
    if dst not in(Format.F16,Format.BF16):raise ValueError('narrow32 destination')
    v=decode(bits,Format.F32)
    if v.kind=='nan':return canonical_nan(dst),NV if v.signaling else 0
    if v.kind=='inf':return (v.sign<<15)|(0x7c00 if dst==Format.F16 else 0x7f80),0
    return encode((-1 if v.sign else 1)*v.magnitude,dst,v.sign)

def gemm(a,b,fmt,initial=None,return_flags=False):
    fmt=Format(fmt); m=len(a); k=len(a[0]); n=len(b[0]); aw=ACC_WIDTHS[fmt]
    result=[[0 for _ in range(n)] for _ in range(m)] if initial is None else [row[:] for row in initial]
    flags=[[0 for _ in range(n)]for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for t in range(k):
                value,fl,_=operate(a[i][t],b[t][j],result[i][j],fmt,mask=1)
                result[i][j]=value&((1<<aw)-1);flags[i][j]|=fl
    return (result,flags)if return_flags else result
