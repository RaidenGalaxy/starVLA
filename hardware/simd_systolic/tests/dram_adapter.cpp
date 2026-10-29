#include "dram_adapter.hpp"
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
int main(int argc,char**argv){
 try{
  if(argc!=3)return 2;std::filesystem::create_directories(argv[2]);
  std::multiset<uint64_t> completed;unsigned writes=0;
  SimdDramAdapter dram(argv[1],argv[2],[&](uint64_t address){completed.insert(address);},
    [&](uint64_t){++writes;});
  for(unsigned i=0;i<32;++i)if(!dram.AddTransaction(0x10000000+64*i,false))throw std::runtime_error("missing logical credit");
  if(dram.AddTransaction(0x10001000,false)||dram.AddTransaction(0x10000001,false))throw std::runtime_error("credit/alignment acceptance");
  unsigned cycles=0;
  while(completed.size()!=32){dram.ClockTick();if(++cycles>100000)throw std::runtime_error("fragmented read timeout");}
  for(unsigned i=0;i<32;++i)if(completed.count(0x10000000+64*i)!=1)throw std::runtime_error("logical read address mismatch");
  completed.clear();
  if(!dram.AddTransaction(0x20000000,true)||!dram.AddTransaction(0x20000040,true))throw std::runtime_error("write acceptance");
  while(writes!=2){dram.ClockTick();if(++cycles>200000)throw std::runtime_error("write acknowledgement timeout");}
  dram.RequestDrain();
  while(!dram.IsDrained()){dram.ClockTick();if(++cycles>300000)throw std::runtime_error("physical tail drain timeout");}
  if(!dram.GetWriteBursts()||dram.GetWriteBursts()>dram.GetSubmittedWrites())throw std::runtime_error("physical write accounting");
  // Two requests for the same block must each complete, even when DRAMSim3
  // merges them into one actual DRAM read command.
  dram.AddTransaction(0x30000000,false);dram.AddTransaction(0x30000000,false);
  while(!dram.IsDrained()){dram.ClockTick();if(++cycles>400000)throw std::runtime_error("duplicate-address callback timeout");}
  if(completed.count(0x30000000)!=2||dram.GetOutstanding())throw std::runtime_error("merged callback accounting");
  if(dram.GetReadBursts()>dram.GetSubmittedReads())throw std::runtime_error("physical read accounting");
  std::cout<<"A "<<dram.GetBurstBytes()<<' '<<dram.GetSubmittedReads()<<' '<<dram.GetReadBursts()<<' '
    <<dram.GetSubmittedWrites()<<' '<<dram.GetWriteBursts()<<' '<<cycles<<'\n';return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
