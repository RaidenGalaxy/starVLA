#include "VSimdPE.h"
#include "verilated.h"
#include "wide.hpp"
#include "ports.hpp"
#include "coverage.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <array>
#include <deque>
#include <algorithm>
#ifndef SEED_LATENCY
#define SEED_LATENCY 1
#endif
#ifndef FIXED_STATE_COMMIT
#define FIXED_STATE_COMMIT 0
#endif
#ifndef STATE_COMMIT_LATENCY
#define STATE_COMMIT_LATENCY 0
#endif
#ifndef REGISTERED_CONTEXT_ADMISSION
#define REGISTERED_CONTEXT_ADMISSION 0
#endif
#ifndef PE_RESPONSE_CREDITS
#define PE_RESPONSE_CREDITS 0
#endif
struct Request {
 unsigned fmt,op,mask,ctx,clear,tag;uint64_t a,b;
 std::string expected;unsigned expectedFlags=0;bool hasExpected=false;
 std::string seed;unsigned seedFlags=0;bool hasSeed=false,seeded=false;
};
int main(int argc,char**argv){
 try{
  Verilated::commandArgs(argc,argv);if(argc<2||argc>3)return 2;
  bool noStall=argc==3&&std::string(argv[2])=="--no-stall";
  bool stateStress=argc==3&&std::string(argv[2])=="--state-stress";
  bool resetStress=argc==3&&std::string(argv[2])=="--reset-stress";
  std::ifstream f(argv[1]);if(!f)throw std::runtime_error("cannot open PE requests");
  std::vector<Request> requests;std::string line;
  while(std::getline(f,line)){
   if(line.empty())continue;
   Request q;std::string a,b,mask;std::istringstream fields(line);
   if(!(fields>>q.fmt>>q.op>>a>>b>>mask>>q.ctx>>q.clear>>q.tag))throw std::runtime_error("invalid PE request");
   q.a=std::stoull(a,nullptr,16);q.b=std::stoull(b,nullptr,16);q.mask=std::stoul(mask,nullptr,16);
   if(fields>>q.expected){
    if(!(fields>>q.expectedFlags)||q.expected.size()>128)throw std::runtime_error("invalid PE state oracle");
    q.expected=std::string(128-q.expected.size(),'0')+q.expected;q.hasExpected=true;
    if(fields>>q.seed){
     if(!(fields>>q.seedFlags)||q.seed.size()>128)throw std::runtime_error("invalid PE preload seed");
     q.hasSeed=true;
    }
   }
   if(stateStress&&FIXED_STATE_COMMIT&&!q.hasExpected)throw std::runtime_error("fixed commit state check needs independent expected values");
   requests.push_back(q);
  }
  VSimdPE d;SimdCoverageSave coverage;d.clock=0;d.reset=1;d.io_seed_valid=0;d.io_in_valid=0;d.io_out_ready=0;d.io_readCtx=0;d.io_readEnable=0;
  for(unsigned i=0;i<4;++i){d.clock=0;d.eval();d.clock=1;d.eval();}d.reset=0;
  if(resetStress){
   if(REGISTERED_CONTEXT_ADMISSION&&PE_RESPONSE_CREDITS<2)
    throw std::runtime_error("admission reset stress requires a registered HF PE");
   for(unsigned phase=0;phase<2;++phase){
    unsigned accepted=0;
    unsigned target=REGISTERED_CONTEXT_ADMISSION?(phase==0?PE_RESPONSE_CREDITS:2):(phase==0?2:1);
    // First fill the reserved response queue under backpressure. Then reset
    // a new same-context request while it is still in the admission buffer.
    unsigned clocks=REGISTERED_CONTEXT_ADMISSION?(phase==0?4*STATE_COMMIT_LATENCY+PE_RESPONSE_CREDITS:2):(phase==0?8:1);
    for(unsigned cycle=0;cycle<clocks;++cycle){
     d.clock=0;d.io_out_ready=0;d.io_in_valid=accepted<target;
     d.io_in_bits_a=accepted+1;d.io_in_bits_b=2;d.io_in_bits_fmt=4;d.io_in_bits_op=1;
     d.io_in_bits_mask=1;d.io_in_bits_clear=1;d.io_in_bits_ctx=phase==0?accepted%16:0;
     d.io_in_bits_tag=accepted;d.eval();
     if(d.io_in_valid&&d.io_in_ready)++accepted;
     d.clock=1;d.eval();
    }
    d.clock=0;d.io_in_valid=0;d.eval();
    if(accepted!=target)throw std::runtime_error("PE reset stress failed to populate in-flight work");
    if(phase==0&&(!d.io_out_valid||d.io_in_ready))
     throw std::runtime_error("PE reset stress did not fill reserved response credits");
    if(phase==1&&(!d.io_pending||d.io_out_valid))
     throw std::runtime_error("PE reset stress did not hold an uncommitted dependency");
    d.reset=1;
    for(unsigned cycle=0;cycle<4;++cycle){d.clock=0;d.eval();d.clock=1;d.eval();}
    d.reset=0;
    for(unsigned cycle=0;cycle<32+STATE_COMMIT_LATENCY;++cycle){
     d.clock=0;d.io_out_ready=1;d.io_readEnable=1;d.io_readCtx=cycle%16;d.eval();
     if(d.io_pending||d.io_out_valid)throw std::runtime_error("PE reset leaked pre-reset work");
     d.clock=1;d.eval();d.clock=0;d.eval();
     if(cycle>=READ_LATENCY&&(wideHex(d.io_readData)!=std::string(128,'0')||d.io_readFlags))
      throw std::runtime_error("PE reset did not clear observable context state");
    }
    std::cout<<"Z "<<phase<<' '<<accepted<<'\n';
   }
   d.io_readEnable=0;
  }
  std::array<std::string,16> state;state.fill(std::string(128,'0'));
  std::array<unsigned,16> stateFlags{};unsigned address=0;bool addressValid=false;
  uint64_t observations=0,seeds=0,commits=0,disabledReads=0;
  struct SeedSnapshot {uint64_t due;unsigned ctx,flags;std::string data;};
  std::deque<SeedSnapshot> seedQueue;
  std::deque<SeedSnapshot> commitQueue;
  // Predict the admission buffer from accepted input transactions and the
  // published latency contract. Neither output handshakes nor internal DUT
  // signals supply the issue or state-commit dates in this oracle.
  std::array<uint64_t,16> scheduledCommit{};
  std::array<bool,16> hasScheduledCommit{};
  struct ClearSnapshot {uint64_t due;unsigned ctx;};
  std::deque<ClearSnapshot> clearQueue;
  uint64_t lastScheduledIssue=0,queuedAdmissions=0,maxAdmissionWait=0;
  bool safeAdmissionClear=false;
  auto monitoredEdge=[&](){
   std::string expected=addressValid?state[address]:std::string(128,'0');
   unsigned expectedFlags=addressValid?stateFlags[address]:0;
   if(FIXED_STATE_COMMIT){
    if(!commitQueue.empty()&&commitQueue.front().due==observations){auto item=commitQueue.front();commitQueue.pop_front();
     state[item.ctx]=item.data;stateFlags[item.ctx]|=item.flags;++commits;}
   }else if(d.io_out_valid&&d.io_out_ready){unsigned ctx=d.io_out_bits_ctx;
    state[ctx]=wideHex(d.io_out_bits_data);stateFlags[ctx]|=d.io_out_bits_flags;++commits;}
   if(d.io_in_valid&&d.io_in_ready&&d.io_in_bits_clear&&
      (!REGISTERED_CONTEXT_ADMISSION||safeAdmissionClear))stateFlags[d.io_in_bits_ctx]=0;
   while(!clearQueue.empty()&&clearQueue.front().due==observations){
    stateFlags[clearQueue.front().ctx]=0;clearQueue.pop_front();}
   if((!commitQueue.empty()&&commitQueue.front().due<observations)||
      (!clearQueue.empty()&&clearQueue.front().due<observations))
    throw std::runtime_error("PE admission calendar skipped a predicted state update");
   if(d.io_seed_valid){seedQueue.push_back({observations+SEED_LATENCY-1,unsigned(d.io_seed_bits_ctx),
     unsigned(d.io_seed_bits_flags),wideHex(d.io_seed_bits_data)});++seeds;}
   if(!seedQueue.empty()&&seedQueue.front().due==observations){auto item=seedQueue.front();seedQueue.pop_front();
    state[item.ctx]=item.data;stateFlags[item.ctx]=item.flags;}
   if(d.io_readEnable){address=d.io_readCtx;addressValid=true;}else ++disabledReads;
   d.clock=1;d.eval();d.clock=0;d.eval();
   if(READ_LATENCY==0){expected=state[d.io_readCtx];expectedFlags=stateFlags[d.io_readCtx];}
   if(wideHex(d.io_readData)!=expected||d.io_readFlags!=expectedFlags)
    throw std::runtime_error("live PE state observation differs at clock "+std::to_string(observations));
   ++observations;
  };
  if(stateStress){
   // Read while seeding all 512 bits, including the first initialization of
   // a context and disabled read-address clocks. No arithmetic is pending.
   for(unsigned cycle=0;cycle<64;++cycle){
    d.clock=0;d.io_readCtx=(cycle*7)%16;d.io_readEnable=cycle%5!=2;
    d.io_seed_valid=cycle<48&&cycle%3==0;d.io_seed_bits_ctx=cycle/3;
    d.io_seed_bits_flags=cycle/3;
    for(unsigned word=0;word<16;++word)d.io_seed_bits_data[word]=0x9e3779b9U*(cycle+word+1);
    d.eval();monitoredEdge();
   }
   d.io_seed_valid=0;
  }
  std::mt19937 rng(822);std::size_t sent=0,received=0;std::vector<uint64_t> issue(requests.size());
  bool held=false;std::string heldData;unsigned heldTag=0,heldFlags=0,heldOverflow=0;
  unsigned seedWait=0;
  for(uint64_t cycle=0;received<requests.size();++cycle){
   if(cycle>1000+requests.size()*100)throw std::runtime_error("PE timeout");
   d.clock=0;d.io_out_ready=noStall||(cycle%137>=65 && rng()%4!=0);
   if(stateStress){d.io_readCtx=(cycle*7)%16;d.io_readEnable=cycle%5!=2;}
   d.io_seed_valid=0;
   bool preload=sent<requests.size()&&requests[sent].hasSeed&&!requests[sent].seeded;
   if(preload&&sent==received&&!d.io_pending){
    auto& x=requests[sent];d.io_seed_valid=1;d.io_seed_bits_ctx=x.ctx;
    d.io_seed_bits_flags=x.seedFlags;setWide(d.io_seed_bits_data,x.seed);
    x.seeded=true;seedWait=SEED_LATENCY+2;
   }
   d.io_in_valid=sent<requests.size()&&!preload&&!seedWait;
   if(d.io_in_valid){auto &x=requests[sent];d.io_in_bits_a=x.a;d.io_in_bits_b=x.b;
    d.io_in_bits_fmt=x.fmt;d.io_in_bits_op=x.op;d.io_in_bits_mask=x.mask;
    d.io_in_bits_ctx=x.ctx;d.io_in_bits_clear=x.clear;d.io_in_bits_tag=x.tag;
   }
   d.eval();
   if(held&&(!d.io_out_valid||d.io_out_bits_tag!=heldTag||wideHex(d.io_out_bits_data)!=heldData||
      d.io_out_bits_flags!=heldFlags||d.io_out_bits_overflow!=heldOverflow))throw std::runtime_error("stalled PE result changed");
   held=d.io_out_valid&&!d.io_out_ready;
   if(held){heldData=wideHex(d.io_out_bits_data);heldTag=d.io_out_bits_tag;heldFlags=d.io_out_bits_flags;heldOverflow=d.io_out_bits_overflow;}
   bool accept=d.io_in_valid&&d.io_in_ready;
   if(d.io_out_valid&&d.io_out_ready){
    if(d.io_out_bits_illegal)throw std::runtime_error("illegal PE operation");
    std::cout<<"O "<<d.io_out_bits_tag<<' '<<wideHex(d.io_out_bits_data)<<' '
     <<unsigned(d.io_out_bits_flags)<<' '<<d.io_out_bits_overflow<<' '<<issue.at(d.io_out_bits_tag)<<' '<<cycle<<'\n';++received;
    std::cout<<"U "<<d.io_out_bits_tag<<' '<<unsigned(d.io_out_bits_ctx)<<'\n';
   }
   safeAdmissionClear=false;
   if(accept){auto &x=requests[sent];issue.at(x.tag)=cycle;
    std::cout<<"J "<<x.tag<<' '<<cycle<<'\n';
    // Oracle values predict observable state; they never drive DUT operands
    // or results. Fixed RF commit is independent of external output ready.
    if(stateStress&&FIXED_STATE_COMMIT){
     uint64_t due=observations+STATE_COMMIT_LATENCY;
     if(REGISTERED_CONTEXT_ADMISSION){
      // A captured request can issue on the next clock. Reusing a context
      // waits until two clocks after its predecessor's state commit: one
      // releases pending, the other refreshes the initialization snapshot.
      if(lastScheduledIssue>observations)
       throw std::runtime_error("PE accepted into an occupied admission buffer");
      safeAdmissionClear=!hasScheduledCommit[x.ctx]||scheduledCommit[x.ctx]<observations;
      uint64_t predictedIssue=observations+1;
      if(hasScheduledCommit[x.ctx])predictedIssue=std::max(predictedIssue,scheduledCommit[x.ctx]+2);
      uint64_t wait=predictedIssue-observations-1;
      if(wait){++queuedAdmissions;maxAdmissionWait=std::max(maxAdmissionWait,wait);}
      lastScheduledIssue=predictedIssue;
      due=predictedIssue+STATE_COMMIT_LATENCY-1;
      scheduledCommit[x.ctx]=due;hasScheduledCommit[x.ctx]=true;
      if(x.clear)clearQueue.push_back({predictedIssue,x.ctx});
     }
     commitQueue.push_back({due,x.ctx,x.expectedFlags,x.expected});
    }
    ++sent;}
   if(stateStress)monitoredEdge();else{d.clock=1;d.eval();}
   if(seedWait)--seedWait;
  }
  d.clock=0;d.io_in_valid=0;d.io_out_ready=1;d.eval();
  if(d.io_pending)throw std::runtime_error("PE pending state did not drain");
  if(stateStress&&(!commitQueue.empty()||!clearQueue.empty()))throw std::runtime_error("predicted PE commits did not drain");
  for(unsigned ctx=0;ctx<16;++ctx){d.io_readCtx=ctx;d.io_readEnable=1;
   for(unsigned i=0;i<READ_LATENCY;++i){d.clock=0;d.eval();d.clock=1;d.eval();d.clock=0;d.eval();}
   d.io_readEnable=0;d.eval();
   std::cout<<"P "<<ctx<<' '<<wideHex(d.io_readData)<<' '<<unsigned(d.io_readFlags)<<'\n';}
  if(stateStress)std::cout<<"Q "<<observations<<' '<<seeds<<' '<<commits<<' '<<disabledReads<<'\n';
  if(stateStress&&REGISTERED_CONTEXT_ADMISSION)std::cout<<"A "<<queuedAdmissions<<' '<<maxAdmissionWait<<'\n';
  simdWriteCoverage();d.final();return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
