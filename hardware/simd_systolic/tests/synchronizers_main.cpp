#include "VSynchronizerProbe.h"
#include "verilated.h"
#include <iostream>
#include <stdexcept>
int main(int argc,char**argv){
 try{
  Verilated::commandArgs(argc,argv);VSynchronizerProbe d;
  d.clock=0;d.externalReset=0;d.data=1;d.eval();
  auto edge=[&](){d.clock=1;d.eval();d.clock=0;d.eval();};
  for(unsigned repeat=0;repeat<32;++repeat){
   d.externalReset=1;d.eval();
   if(!d.releasedReset||d.toggle)throw std::runtime_error("reset did not assert without a clock edge");
   d.externalReset=0;d.data=1;d.eval();
   if(!d.releasedReset||d.toggle)throw std::runtime_error("asynchronous reset release");
   edge();if(!d.releasedReset||d.toggle)throw std::runtime_error("synchronizer released before two edges");
   edge();if(d.releasedReset||!d.toggle)throw std::runtime_error("synchronizer latency is not two edges");
   d.data=0;d.eval();edge();if(!d.toggle)throw std::runtime_error("toggle bypasses first stage");
   edge();if(d.toggle)throw std::runtime_error("toggle did not cross both stages");
  }
  d.final();std::cout<<"{\"passed\":true,\"reset_releases\":32,\"toggle_transitions\":64,\"asynchronous_assertion\":true,\"release_edges\":2}\n";
  return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
