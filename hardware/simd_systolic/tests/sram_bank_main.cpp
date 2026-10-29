#include "VSimdSramBank.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) {
  Verilated::commandArgs(argc,argv);VSimdSramBank d;
  std::array<std::array<uint64_t,2>,2048> expected{};
  unsigned writes=0,reads=0,rmw=0,physicalWriteBytes=0;
  auto tick=[&](){d.clock=0;d.eval();d.clock=1;d.eval();d.clock=0;d.eval();};
  auto require=[](bool good){if(!good)throw std::runtime_error("SRAM adapter mismatch");};
  d.reset=1;d.readEnable=0;d.writeValid=0;tick();d.reset=0;tick();
  auto write=[&](unsigned address,uint64_t lo,uint64_t hi,unsigned mask){
    d.writeAddress=address;d.writeData[0]=lo;d.writeData[1]=lo>>32;
    d.writeData[2]=hi;d.writeData[3]=hi>>32;d.writeStrobe=mask;d.writeValid=1;d.eval();
    require(d.writeReady);bool partial=false;
    for(unsigned w=0;w<2;++w){unsigned sub=(mask>>(8*w))&255;partial|=sub!=0&&sub!=255;}
    unsigned expectedBytes=8*(((mask&255)!=0)+(((mask>>8)&255)!=0));
    unsigned actualBytes=d.writeBytes;
    tick();d.writeValid=0;d.eval();
    if(partial){require(d.busy&&!d.readReady&&!d.writeReady);++rmw;actualBytes+=d.writeBytes;tick();}
    require(actualBytes==expectedBytes);physicalWriteBytes+=actualBytes;
    require(!d.busy&&d.writeDone);
    for(unsigned byte=0;byte<16;++byte)if(mask>>byte&1){
      auto value=byte<8?lo:hi;unsigned shift=(byte%8)*8;
      expected[address][byte/8]=(expected[address][byte/8]&~(uint64_t(255)<<shift))|(value&(uint64_t(255)<<shift));
    }
    tick();++writes;
  };
  auto read=[&](unsigned address){
    d.readAddress=address;d.readEnable=1;d.eval();require(d.readReady);tick();
    require(d.readValid);
    for(unsigned w=0;w<2;++w){uint64_t value=d.readData[w*2]|(uint64_t(d.readData[w*2+1])<<32);require(value==expected[address][w]);}
    d.readEnable=0;tick();require(!d.readValid);++reads;
  };
  for(unsigned a=0;a<2048;++a)write(a,0x1234567800000000ULL^a,0xfedcba9800000000ULL^uint64_t(a*7),65535);
  for(unsigned a=2048;a-->0;)read(a);
  for(unsigned a: {0u,1u,1023u,1024u,2047u})for(unsigned mask:
      {0u,1u,128u,255u,256u,65280u,65535u,21845u,43690u,65281u}){
    write(a,0xdeadbeef12345678ULL,0x0506070801020304ULL,mask);read(a);
  }
  // An unacknowledged transaction may be cancelled by reset; completed
  // locations retain data, and the adapter must become usable again.
  d.writeAddress=12;d.writeStrobe=1;d.writeValid=1;tick();d.writeValid=0;
  require(d.busy);d.reset=1;tick();require(!d.busy&&!d.readValid&&!d.writeDone);
  d.reset=0;tick();read(11);read(13);
  write(12,17,23,65535);read(12);
  std::cout<<"{\"passed\":true,\"writes\":"<<writes<<",\"reads\":"<<reads
           <<",\"rmw_transactions\":"<<rmw<<",\"physical_write_bytes\":"<<physicalWriteBytes<<",\"depth\":2048,\"width\":128,\"reset_retains_completed_words\":true}\n";
}
