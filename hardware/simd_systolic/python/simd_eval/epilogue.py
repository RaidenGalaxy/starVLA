"""Exact reference for the optional FP32 output scale and destination cast."""
from fractions import Fraction
from .reference import Format,Op,WIDTHS,ACC_WIDTHS,ENCODINGS,decode,encode,canonical_nan,round_even,signed,fp_operation,NV,OF,NX


def floating_cast(bits,source,destination):
    value=decode(bits,source)
    if value.kind=='nan':return canonical_nan(destination),NV if value.signaling else 0
    if value.kind=='inf':
        exponent,mantissa=ENCODINGS[destination]
        return(value.sign<<(exponent+mantissa))|(((1<<exponent)-1)<<mantissa),0
    return encode((-1 if value.sign else 1)*value.magnitude,destination,value.sign)


def fold32(bits,source):
    source=Format(source)
    if source<Format.F4:return encode(Fraction(signed(bits,ACC_WIDTHS[source])),Format.F32)
    if source==Format.F64:return floating_cast(bits,Format.F64,Format.F32)
    return bits&0xffffffff,0


def convert32(bits,destination,zero_point=0):
    destination=Format(destination);value=decode(bits,Format.F32)
    if destination<Format.F4:
        width=WIDTHS[destination];lower=-(1<<(width-1));upper=(1<<(width-1))-1
        invalid=value.kind=='nan';inexact=False
        if invalid:integer=0
        elif value.kind=='inf':integer=(-1 if value.sign else 1)*(1<<65)
        else:
            integer,inexact=round_even(value.magnitude)
            if value.sign:integer=-integer
        integer+=signed(zero_point,32);saturated=not lower<=integer<=upper
        clipped=max(lower,min(upper,integer))&((1<<width)-1)
        return clipped,NV if invalid or saturated else NX if inexact else 0,saturated
    if destination==Format.F4:
        if value.kind=='nan':return 0,NV,False
        if value.kind=='inf':return(value.sign<<3)|7,OF|NX,True
        result,flags=encode((-1 if value.sign else 1)*value.magnitude,Format.F4,value.sign)
        saturated=value.magnitude>6
        return result,flags|(OF|NX if saturated else 0),saturated
    result,flags=floating_cast(bits,Format.F32,destination)
    return result,flags,False


def apply(accumulator,source,destination,scale_bits=0x3f800000,zero_point=0):
    value,flags=fold32(accumulator,source)
    scaled,multiply_flags=fp_operation(value,scale_bits,0,Format.F32,Op.MUL)
    result,cast_flags,saturated=convert32(scaled,destination,zero_point)
    return result,flags|multiply_flags|cast_flags,saturated


def output_configuration(caps,format=None,scale='1',zero_point=0):
    """Optional workload output contract; the scale is rounded to FP32 once."""
    if format is None:
        if Fraction(scale)!=1 or zero_point:
            raise ValueError('output scale/zero point require --output-format')
        return None
    if not caps.get('output_epilogue'):
        raise ValueError('build has no device output epilogue')
    destination=Format[format]
    if not -(1<<31)<=zero_point<(1<<31):
        raise ValueError('output zero point must fit signed INT32')
    if destination>=Format.F4 and zero_point:
        raise ValueError('floating outputs require zero point 0')
    scale_bits,_=encode(Fraction(scale),Format.F32)
    decoded=decode(scale_bits,Format.F32)
    if decoded.kind!='finite' or not decoded.magnitude:
        raise ValueError('workload output scale must be finite and nonzero after FP32 rounding')
    return dict(output_format=destination.name,output_scale_bits=scale_bits,
                output_scale_exact=str((-1 if decoded.sign else 1)*decoded.magnitude),
                output_zero_point=zero_point,output_arithmetic='FP32_scale_then_RNE_cast',
                output_error_domain='reconstructed_original_values')


def finish_matrix(matrix,source,configuration):
    if configuration is None:return matrix,0,0
    result=[];flags=saturations=0
    for row in matrix:
        values=[]
        for value in row:
            code,status,clipped=apply(value,source,Format[configuration['output_format']],
                configuration['output_scale_bits'],configuration['output_zero_point'])
            values.append(code);flags|=status;saturations+=int(clipped)
        result.append(values)
    return result,flags,saturations


def packed_row(values,width):
    return sum((value&((1<<width)-1))<<(column*width)
               for column,value in enumerate(values)).to_bytes((len(values)*width+7)//8,'little')


def read_matrix(written,address,stride,m,n,width):
    result=[]
    for row in range(m):
        raw=sum(written[address+row*stride+byte]<<(byte*8)for byte in range((n*width+7)//8))
        result.append([(raw>>(column*width))&((1<<width)-1)for column in range(n)])
    return result
