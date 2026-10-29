#include "VSystolicArray.h"
#include "verilated.h"
#include "coverage.hpp"
#include "ports.hpp"
#include "wide.hpp"
#include "assertions.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <deque>
#ifndef SIMULATION_THREADS
#define SIMULATION_THREADS 1
#endif
int main(int argc,char**argv){
 try {
 if(argc<2)return 2;Verilated::commandArgs(argc,argv);
 simdConfigureAssertions();
 Verilated::threadContextp()->threads(SIMULATION_THREADS);VSystolicArray d;SimdCoverageSave coverage;
 if(d.threads()!=SIMULATION_THREADS){std::cerr<<"generated root thread count differs from simulation configuration\n";return 2;}
 d.clock=0;d.reset=1;d.io_seed_valid=0;d.io_inject=0;d.io_readEnable=0;uint64_t cycle=0;
 struct PendingRead{unsigned tag;uint64_t due;};std::deque<PendingRead> pending;
 auto observe=[&](unsigned tag){
  std::cout<<"R "<<tag<<' '<<wideHex(d.io_readData)<<' '<<unsigned(d.io_readFlags)
   <<' '<<cycle<<' '<<d.io_macs<<' '<<d.io_completions<<'\n';
 };
 auto tick=[&](){d.clock=0;simdEval(d);d.clock=1;simdEval(d);cycle++;d.clock=0;simdEval(d);
  if(!pending.empty()&&pending.front().due==cycle){observe(pending.front().tag);pending.pop_front();}
 };
 for(int i=0;i<4;i++)tick();d.reset=0;cycle=0;
 std::ifstream f(argv[1]);if(!f)throw std::runtime_error("trace file not found");std::string line;
 while(std::getline(f,line)){
  std::istringstream s(line);char command;s>>command;if(!s)continue;
  d.io_inject=0;
  if(command!='P'){d.io_readEnable=0;while(!pending.empty())tick();}
  if(command=='I'){
   unsigned fmt,ctx,clear,tag;std::string rm;s>>fmt>>ctx>>clear>>tag>>rm;
   d.io_fmt=fmt;d.io_ctx=ctx;d.io_clear=clear;d.io_tag=tag;d.io_rowMask=std::stoul(rm,nullptr,16);
   for(int i=0;i<COLS;i++){std::string x;s>>x;setMask(d,i,std::stoul(x,nullptr,16));}
   for(int i=0;i<ROWS;i++){std::string x;s>>x;setA(d,i,std::stoull(x,nullptr,16));}
   for(int i=0;i<COLS;i++){std::string x;s>>x;setB(d,i,std::stoull(x,nullptr,16));}
   d.io_inject=1;tick();d.io_inject=0;
  }else if(command=='W'){unsigned n;s>>n;while(n--)tick();}
  else if(command=='T'){
   unsigned tag;s>>tag;
   std::cout<<"T "<<tag<<' '<<cycle<<' '<<d.io_macs<<' '<<d.io_completions<<'\n';
  }
  else if(command=='E'){
   unsigned tag;s>>tag;simdEval(d);
   std::cout<<"E "<<tag<<' '<<cycle<<' '<<d.io_macs<<' '<<d.io_completions
    <<' '<<d.io_overflowEvents<<' '<<d.io_exceptionOps<<'\n';
  }
  else if(command=='S'){
   unsigned ctx,row,col,flags;std::string data;s>>ctx>>row>>col>>flags>>data;
   simdEval(d);if(d.io_busy)return 4;
   d.io_seed_bits_row=row;d.io_seed_bits_col=col;d.io_seed_bits_value_ctx=ctx;
   d.io_seed_bits_value_flags=flags;setWide(d.io_seed_bits_value_data,data);
   d.io_seed_valid=1;tick();d.io_seed_valid=0;
   // Current F2/HF seed writes take at most two clocks, followed by the
   // registered array busy release. Keep initialization fully drained.
   tick();tick();
  }
  else if(command=='D'){
   tick();uint64_t timeout=0;while(d.io_busy){tick();if(++timeout>100000)return 3;}
  }else if(command=='R'||command=='P'){
   unsigned ctx,row,col,tag;s>>ctx>>row>>col>>tag;
   d.io_readCtx=ctx;d.io_readRow=row;d.io_readCol=col;d.io_readEnable=1;
   if(command=='P'){
    if(READ_LATENCY){pending.push_back({tag,cycle+READ_LATENCY});tick();}
    else {simdEval(d);observe(tag);tick();}
    continue;
   }
   for(unsigned i=0;i<READ_LATENCY;++i)tick();simdEval(d);d.io_readEnable=0;
   std::cout<<"R "<<tag<<' '<<wideHex(d.io_readData)<<' '<<unsigned(d.io_readFlags)
    <<' '<<cycle<<' '<<d.io_macs<<' '<<d.io_completions<<'\n';
  }
 }
 d.io_readEnable=0;while(!pending.empty())tick();
 simdWriteCoverage();d.final();return 0;
 }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
