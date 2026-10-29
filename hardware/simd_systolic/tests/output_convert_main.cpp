#include "VSimdOutputConvert.h"
#include "verilated.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char**argv){
 try{
  Verilated::commandArgs(argc,argv);if(argc!=2)return 2;VSimdOutputConvert d;
  std::ifstream input(argv[1]);uint64_t value,expected;unsigned wide,format,zero,flags,saturated;unsigned count=0;
  while(input>>std::hex>>value>>wide>>format>>zero>>expected>>flags>>saturated){
   d.value=value;d.wide_source=wide;d.format=format;d.zero_point=zero;d.eval();
   if(d.result!=expected||d.flags!=flags||unsigned(d.saturated)!=saturated){
    std::cerr<<"vector "<<std::dec<<count<<" input "<<std::hex<<value<<" format "<<format
     <<" expected "<<expected<<' '<<flags<<' '<<saturated<<" actual "<<d.result<<' '
     <<unsigned(d.flags)<<' '<<unsigned(d.saturated)<<'\n';return 1;
   }++count;
  }
  d.final();std::cout<<"{\"passed\":true,\"vectors\":"<<std::dec<<count<<"}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
