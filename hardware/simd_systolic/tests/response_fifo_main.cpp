#include "VResponseFifoProbe.h"
#include "verilated.h"
#include <array>
#include <deque>
#include <iostream>
#include <random>
#include <stdexcept>

int main(int argc,char**argv) {
 try {
  Verilated::commandArgs(argc,argv);VResponseFifoProbe d;
  std::array<unsigned,4> depths{10,11,14,15};
  std::array<std::deque<uint32_t>,4> expected;
  std::array<unsigned,4> serial{},emptyWrites{},emptyReady{},fullReplace{},lastReplace{},holds{},accepted{};
  std::mt19937 rng(5281);unsigned resets=0;
  for(unsigned cycle=0;cycle<20000;++cycle) {
   bool reset=cycle<4||(cycle>=6000&&cycle<6004)||(cycle>=12000&&cycle<12004)||(cycle>=18000&&cycle<18004);
   if(cycle==6000||cycle==12000||cycle==18000)++resets;
   d.clock=0;d.reset=reset;d.write=0;d.read=0;
   for(unsigned bank=0;bank<4;++bank) {
    auto &q=expected[bank];bool read=false,write=false;
    if(!reset) {
     if(cycle<20)write=q.size()<depths[bank];
     else if(cycle<532)read=write=true; // full-queue replacement across wraps
     else if(cycle<548)read=!q.empty();
     else if(cycle==548)write=true;
     else if(cycle<1061)read=write=true; // simultaneous last-entry replacement
     else {
      read=cycle%137>=65&&rng()%4!=0;
      write=(q.size()<depths[bank]||read)&&rng()%5!=0;
     }
    }
    d.read|=unsigned(read)<<bank;d.write|=unsigned(write)<<bank;
    d.data[bank]=(bank<<28)|serial[bank];
   }
   d.eval();
   for(unsigned bank=0;bank<4;++bank) {
    auto &q=expected[bank];bool ready=(d.read>>bank)&1,read=ready&&!q.empty(),write=(d.write>>bank)&1;
    if(!reset&&!q.empty()&&d.value[bank]!=q.front())throw std::runtime_error("head differs before edge");
    if(reset){q.clear();continue;}
    if(write&&q.empty())++emptyWrites[bank];
    if(ready&&q.empty())++emptyReady[bank];
    if(write&&read&&q.size()==depths[bank])++fullReplace[bank];
    if(write&&read&&q.size()==1)++lastReplace[bank];
    if(!read&&!q.empty())++holds[bank];
    if(read){if(q.empty())throw std::runtime_error("invalid test read");q.pop_front();}
    if(write){q.push_back(d.data[bank]);++serial[bank];++accepted[bank];}
   }
   d.clock=1;d.eval();d.clock=0;d.eval();
   for(unsigned bank=0;bank<4;++bank)
    if(!expected[bank].empty()&&d.value[bank]!=expected[bank].front())
     throw std::runtime_error("prefetched head differs after edge at clock "+std::to_string(cycle));
  }
  unsigned transactions=0;
  for(unsigned bank=0;bank<4;++bank) {
   if(emptyWrites[bank]<4||emptyReady[bank]==0||fullReplace[bank]<512||lastReplace[bank]<512||holds[bank]==0)
    throw std::runtime_error("missing FIFO boundary coverage");
   transactions+=accepted[bank];
  }
  d.final();std::cout<<"{\"passed\":true,\"depths\":[10,11,14,15],\"clocks\":20000,\"accepted\":"<<transactions
   <<",\"inflight_resets\":"<<resets<<",\"full_replace_per_bank_min\":512,\"last_replace_per_bank_min\":512,\"held_head_checked\":true,\"empty_ready_checked\":true}\n";
  return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
