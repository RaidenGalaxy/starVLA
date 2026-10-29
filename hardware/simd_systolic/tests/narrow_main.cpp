#include "VSimdNarrow32.h"
#include "verilated.h"
#include <cstdint>
#include <fstream>
#include <iostream>
int main(int argc,char**argv){
 if(argc!=2)return 2;Verilated::commandArgs(argc,argv);VSimdNarrow32 d;
 std::ifstream in(argv[1]);uint64_t value,result;unsigned bf16,flags;unsigned count=0;
 while(in>>std::hex>>value>>bf16>>result>>flags){
  d.value=value;d.bf16=bf16;d.eval();++count;
  if(d.result!=result||d.flags!=flags){
   std::cerr<<"conversion "<<std::hex<<value<<" bf16="<<bf16<<" expected="<<result<<":"<<flags
            <<" actual="<<d.result<<":"<<unsigned(d.flags)<<'\n';return 1;
  }
 }
 d.final();std::cout<<"passed "<<count<<" narrowing vectors\n";return 0;
}
