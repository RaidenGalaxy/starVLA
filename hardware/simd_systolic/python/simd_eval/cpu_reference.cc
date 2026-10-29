// SPDX-License-Identifier: Apache-2.0
// Independent host FMA oracle. Every output retains the original K order.
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559,"IEEE binary32 host required");
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#pragma STDC FENV_ACCESS ON
static float from_bits(uint32_t bits){float value;std::memcpy(&value,&bits,4);return value;}
static uint32_t bits_of(float value){uint32_t bits;std::memcpy(&bits,&value,4);return bits;}
static float expand(uint64_t code,int format){
 if(format==5){
  static const uint32_t values[]={0,0x3f000000,0x3f800000,0x3fc00000,0x40000000,0x40400000,0x40800000,0x40c00000};
  return from_bits(values[code&7]|((code&8)?0x80000000:0));
 }
 if(format==6){code=(code&255)<<8;format=7;}
 if(format==9)return from_bits(uint32_t(code));
 if(format==8)return from_bits(uint32_t(code&65535)<<16);
 uint32_t sign=(code&0x8000)<<16,exponent=(code>>10)&31,fraction=code&1023;
 if(exponent==31)return from_bits(sign|0x7f800000|(fraction<<13));
 if(exponent)return from_bits(sign|((exponent+112)<<23)|(fraction<<13));
 if(!fraction)return from_bits(sign);
 // All FP16 subnormals are exact normal FP32 values.
 int shift=0;while(!(fraction&1024)){fraction<<=1;++shift;}
 return from_bits(sign|(uint32_t(113-shift)<<23)|((fraction&1023)<<13));
}
extern "C" int simd_fma_gemm(const uint64_t* a,const uint64_t* b,const uint32_t* initial,
 uint32_t* output,uint64_t m,uint64_t n,uint64_t k,int format,uint32_t* status){
 if(format<5||format>9)return 1;
 std::fenv_t saved;std::fegetenv(&saved);std::fesetround(FE_TONEAREST);
#if defined(__SSE__)
 unsigned control=_mm_getcsr();_mm_setcsr(control&~((1u<<15)|(1u<<6)));
#endif
 for(uint64_t i=0;i<m;++i)for(uint64_t j=0;j<n;++j){
  std::feclearexcept(FE_ALL_EXCEPT);
  float accumulator=initial?from_bits(initial[i*n+j]):0.0f;
  for(uint64_t t=0;t<k;++t)accumulator=std::fma(expand(a[i*k+t],format),expand(b[t*n+j],format),accumulator);
  uint32_t result=bits_of(accumulator);
  if((result&0x7fffffff)>0x7f800000)result=0x7fc00000;
  output[i*n+j]=result;
  int raised=std::fetestexcept(FE_ALL_EXCEPT);
  status[i*n+j]=((raised&FE_INVALID)?16:0)|((raised&FE_DIVBYZERO)?8:0)|((raised&FE_OVERFLOW)?4:0)|
    ((raised&FE_UNDERFLOW)?2:0)|((raised&FE_INEXACT)?1:0);
 }
#if defined(__SSE__)
 _mm_setcsr(control);
#endif
 std::fesetenv(&saved);return 0;
}
