"""Group-wise symmetric quantization with explicitly rounded FP32 scales."""
from fractions import Fraction
from .reference import Format,WIDTHS,decode,encode,round_even


def load_codes(path, fmt):
    """Import actual encoded tensors and FP32 scale bits without requantizing.

    A/B retain the original tensors for workload identity and error reporting;
    qa/qb are unsigned element encodings, not signed numeric values.
    """
    import numpy as np
    data=np.load(path,allow_pickle=False);fmt=Format(fmt)
    if str(data['format'].item())!=fmt.name:raise ValueError('quantized checkpoint format differs from selected format')
    group_size=int(data['group_size'].item())
    if group_size<1:raise ValueError('invalid checkpoint group size')
    a=data['a'];b=data['b'];qa=data['qa'];qb=data['qb'];sa=data['sa_bits'];sb=data['sb_bits']
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0] or min(*a.shape,*b.shape)<1:
        raise ValueError('invalid checkpoint A[M,K]/B[K,N] shapes')
    m,k=a.shape;n=b.shape[1];groups=(k+group_size-1)//group_size
    if qa.shape!=(m,k)or qb.shape!=(k,n)or sa.shape!=(m,groups)or sb.shape!=(groups,n):
        raise ValueError('quantized checkpoint code/scale shapes disagree')
    for array,bits in ((qa,WIDTHS[fmt]),(qb,WIDTHS[fmt]),(sa,32),(sb,32)):
        if array.dtype.kind not in 'ui' or np.any(array<0)or np.any(array>=(1<<bits)):
            raise ValueError('codes/scales must contain unsigned element encodings')
    for bits in (*sa.flat,*sb.flat):
        value=decode(int(bits),Format.F32)
        if value.kind!='finite' or (value.sign and value.magnitude):raise ValueError('scales must be finite nonnegative FP32')
    original=lambda x:[[Fraction(float(v))for v in row]for row in x]
    return original(a),original(b),qa.tolist(),qb.tolist(),sa.tolist(),sb.tolist(),group_size

def quantize(a,b,fmt,group_size):
    fmt=Format(fmt);m=len(a);k=len(a[0]);n=len(b[0]);groups=(k+group_size-1)//group_size
    qa=[[0]*k for _ in range(m)];qb=[[0]*n for _ in range(k)]
    sa=[[0]*groups for _ in range(m)];sb=[[0]*n for _ in range(groups)]
    if fmt not in (Format.I4,Format.I8,Format.F4,Format.F8):raise ValueError('use I4/I8/F4/F8')
    qmax={Format.I4:7,Format.I8:127,Format.F4:6,Format.F8:57344}[fmt]
    def block(values):
        maxabs=max(map(abs,values),default=Fraction(0));scale=maxabs/qmax if maxabs else Fraction(1)
        bits,_=encode(scale,Format.F32)
        if bits==0:bits=1
        decoded=decode(bits,Format.F32)
        if decoded.kind!='finite':raise ValueError('group scale is outside finite FP32 range')
        scale=decoded.magnitude
        def q(v):
            ratio=v/scale
            if fmt<Format.F4:
                result=round_even(abs(ratio))[0]*(-1 if ratio<0 else 1)
                return max(-qmax,min(qmax,result))&((1<<(4 if fmt==Format.I4 else 8))-1)
            return encode(ratio,fmt)[0]
        return [q(v)for v in values],bits
    for g in range(groups):
        lo=g*group_size;hi=min(k,lo+group_size)
        for i in range(m):
            qa[i][lo:hi],sa[i][g]=block(a[i][lo:hi])
        for j in range(n):
            values,sb[g][j]=block([b[t][j]for t in range(lo,hi)])
            for t,v in zip(range(lo,hi),values):qb[t][j]=v
    return qa,qb,sa,sb
