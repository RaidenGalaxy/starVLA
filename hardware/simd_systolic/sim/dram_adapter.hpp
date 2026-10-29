#pragma once
#include <cstdint>
#include <cmath>
#include "dramsim3.h"
#include <algorithm>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

// The RTL transfers aligned 64-byte logical transactions. A DRAM transaction
// may be smaller (fragmentation) or larger (aligned, byte-masked transfer).
// Functional data/byte enables stay in the simulator's backing memory; this
// adapter schedules only actual DRAM timing transactions and acknowledgements.
class SimdDramAdapter {
 struct Request {uint64_t address;unsigned remaining;bool write;};
 struct Fragment {uint64_t address;std::shared_ptr<Request> request;};
 std::unique_ptr<dramsim3::MemorySystem> memory;
 std::deque<Fragment> pending;
 std::map<uint64_t,std::deque<std::shared_ptr<Request>>> readWaiting,writeWaiting;
 std::function<void(uint64_t)> readCallback,writeCallback;
 unsigned burstBytes=0,outstanding=0;
 bool drainRequested=false;
 uint64_t submittedReads=0,submittedWrites=0;
 void complete(uint64_t address,bool write){
  auto&waiting=write?writeWaiting:readWaiting;auto found=waiting.find(address);
  if(found==waiting.end()||found->second.empty())throw std::runtime_error("unmatched physical DRAM callback");
  auto request=found->second.front();found->second.pop_front();
  if(found->second.empty())waiting.erase(found);
  if(!request->remaining)throw std::runtime_error("duplicate physical DRAM callback");
  if(--request->remaining==0){
   --outstanding;(write?writeCallback:readCallback)(request->address);
  }
 }
 void pump(){
  while(!pending.empty()){
   auto fragment=pending.front();auto write=fragment.request->write;
   if(!memory->WillAcceptTransaction(fragment.address,write))break;
   auto&waiting=write?writeWaiting:readWaiting;
   waiting[fragment.address].push_back(fragment.request);pending.pop_front();
   if(!memory->AddTransaction(fragment.address,write))throw std::runtime_error("DRAM rejected an advertised transaction slot");
   if(write)++submittedWrites;else ++submittedReads;
  }
  if(drainRequested)memory->RequestDrain();
 }
 public:
 SimdDramAdapter(const std::string&config,const std::string&output,
   std::function<void(uint64_t)>read,std::function<void(uint64_t)>write)
   :readCallback(std::move(read)),writeCallback(std::move(write)){
  memory.reset(dramsim3::GetMemorySystem(config,output,
    [this](uint64_t address){complete(address,false);},
    [this](uint64_t address){complete(address,true);}));
  const auto bits=memory->GetBusBits()*memory->GetBurstLength();
  if(bits<=0||bits%8)throw std::runtime_error("DRAM burst must contain whole bytes");
  burstBytes=bits/8;
  if(burstBytes<8||burstBytes>4096||(burstBytes&(burstBytes-1)))
    throw std::runtime_error("DRAM burst bytes must be a power of two in [8,4096]");
 }
 bool WillAcceptTransaction(uint64_t address,bool)const{return address%64==0&&outstanding<32;}
 bool AddTransaction(uint64_t address,bool write){
  if(!WillAcceptTransaction(address,write))return false;
  const auto count=std::max(1u,64/burstBytes);
  auto request=std::make_shared<Request>(Request{address,count,write});++outstanding;
  for(unsigned i=0;i<count;++i)pending.push_back(Fragment{
    (address+uint64_t(i)*burstBytes)&~uint64_t(burstBytes-1),request});
  pump();return true;
 }
 void ClockTick(){memory->ClockTick();pump();}
 void RequestDrain(){drainRequested=true;pump();}
 bool IsDrained()const{return pending.empty()&&outstanding==0&&memory->IsDrained();}
 double GetTCK()const{return memory->GetTCK();}
 int GetBusBits()const{return memory->GetBusBits();}
 int GetBurstLength()const{return memory->GetBurstLength();}
 unsigned GetBurstBytes()const{return burstBytes;}
 unsigned GetOutstanding()const{return outstanding;}
 uint64_t GetReadBursts()const{return memory->GetReadBursts();}
 uint64_t GetWriteBursts()const{return memory->GetWriteBursts();}
 uint64_t GetSubmittedReads()const{return submittedReads;}
 uint64_t GetSubmittedWrites()const{return submittedWrites;}
 void PrintStats()const{
  memory->PrintStats();
  std::cout<<"K "<<uint64_t(std::llround(GetTCK()*1000.0))<<' '
   <<GetBusBits()<<' '<<GetBurstLength()<<'\n';
  std::cout<<"P "<<burstBytes<<' '<<submittedReads<<' '<<submittedWrites<<'\n';
 }
};
