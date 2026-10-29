// Independently use the host's exact binary16 conversion for finite values.
#include "VFloatExpansionProbe.h"
#include "verilated.h"
#include <cstdint>
#include <cstring>
#include <iostream>
static uint32_t half(uint16_t code){
 if((code&0x7c00)==0x7c00)return (uint32_t(code&0x8000)<<16)|0x7f800000|(uint32_t(code&0x3ff)<<13);
 _Float16 value;std::memcpy(&value,&code,2);float expanded=static_cast<float>(value);
 uint32_t bits;std::memcpy(&bits,&expanded,4);return bits;
}
int main(int argc,char**argv){
 Verilated::commandArgs(argc,argv);VFloatExpansionProbe dut;unsigned count=0;
 const uint32_t fp4[]={0,0x3f000000,0x3f800000,0x3fc00000,0x40000000,0x40400000,0x40800000,0x40c00000};
 for(unsigned fmt=5;fmt<=8;++fmt)for(unsigned code=0;code<(fmt==5?16u:fmt==6?256u:65536u);++code){
  uint32_t expected=fmt==5?(fp4[code&7]|((code&8)?0x80000000:0)):fmt==6?half(code<<8):fmt==7?half(code):code<<16;
  dut.io_fmt=fmt;dut.io_code=code;dut.eval();
  if(dut.io_value!=expected){std::cerr<<"expansion mismatch fmt="<<fmt<<" code="<<code<<'\n';return 1;}++count;
 }
 dut.final();std::cout<<"{\"passed\":true,\"vectors\":"<<count<<",\"all_encodings\":[\"F4\",\"F8\",\"F16\",\"BF16\"]}\n";
}
