#include "VGemmSystem.h"
#include "verilated.h"
#include "coverage.hpp"
#include "wide.hpp"
#include "assertions.hpp"
#include "ports.hpp"
#include "dram_adapter.hpp"
#include "clocking.hpp"
#ifndef SIMULATION_THREADS
#define SIMULATION_THREADS 1
#endif
#ifndef HAS_NATIVE_PREFETCH
#define HAS_NATIVE_PREFETCH 0
#endif
#ifndef HAS_OUTPUT_EPILOGUE
#define HAS_OUTPUT_EPILOGUE 0
#endif
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <set>
#include <stdexcept>
#include <vector>

struct Descriptor {unsigned fmt,m,n,k,tag;uint64_t a,b,c,as,bs,cs;bool weight4=false,accumulate=false;
 uint64_t scales=0,scaleStride=0;unsigned groupLog2=0;bool grouped=false;uint64_t activationScales=0,activationScaleStride=0;bool prefetch=true;
 bool outputConvert=false;unsigned outputFormat=9;uint32_t outputScale=0x3f800000;int32_t outputZeroPoint=0;};
class Simulation {
 public:
 VGemmSystem d;uint64_t now=0,ct=0,mt=0,dt=0,ch,mh,dp=0,start=0,submitted=0;
 uint64_t readRequests=0,writeRequests=0,writeBytes=0,commandCount=0;
 uint64_t epochWriteRequests=0,epochWriteBytes=0,oldReadBursts=0,oldWriteBursts=0;
 uint64_t descriptorReadStart=0,descriptorWriteStart=0;unsigned epoch=0;
 bool allowPrefetch=true;uint64_t prefetchLoadBytes=0,prefetchCoreBytes=0;
 bool outputConvert=false;unsigned outputFormat=9;uint32_t outputScale=0x3f800000;int32_t outputZeroPoint=0;
 std::string memoryConfiguration,memoryOutput;
 uint64_t coreEdges=0,memoryEdges=0;
 uint64_t readInFlight=0,writeInFlight=0,maxReadInFlight=0,maxWriteInFlight=0;
 uint64_t readDataBeats=0,writeDataBeats=0;
 uint32_t rng;bool commandAccepted=false,timerStarted=false;
 std::map<uint64_t,uint8_t> memory,written;
 std::set<uint64_t> readFaults,writeFaults;bool expectedFailure=false;
 std::deque<uint64_t> reads,writes;
 std::unique_ptr<SimdDramAdapter> dram;
 bool reading=false,writing=false;uint64_t readAddress=0,writeAddress=0;
 unsigned readMask=0,readBeat=0,writeBeat=0;
 Simulation(uint64_t core,uint64_t mem,const std::string& cfg,const std::string& out,unsigned seed)
  :ch(core/2),mh(mem/2),rng(seed),memoryConfiguration(cfg),memoryOutput(out){
  if(!ch||!mh||core%2||mem%2)throw std::runtime_error("even positive clock periods required");
  ct=simdClockPhase("SIMD_CORE_PHASE_PS",core);mt=simdClockPhase("SIMD_MEMORY_PHASE_PS",mem);dt=ct;
  if(d.threads()!=SIMULATION_THREADS)throw std::runtime_error("generated root thread count differs from simulation configuration");
  d.io_coreClock=0;d.io_memoryClock=0;d.io_coreReset=0;d.io_memoryReset=0;
  d.io_command_valid=0;d.io_done_ready=0;d.io_memoryRequest_ready=0;
  d.io_memoryResponse_valid=0;d.io_writeRequest_ready=0;d.io_writeData_ready=0;d.io_writeResponse_valid=0;
#if HAS_SCRATCHPAD
  d.io_scratch_valid=0;d.io_scratchResult_ready=0;
#endif
  if(cfg!="-"){
   std::filesystem::create_directories(out);
   dram.reset(new SimdDramAdapter(cfg,out,[&](uint64_t a){reads.push_back(a);},[&](uint64_t a){writes.push_back(a);}));
   dp=uint64_t(std::llround(dram->GetTCK()*1000));
   if(!dp)throw std::runtime_error("DRAM tCK must be positive");
  }
  std::cout<<"J "<<core<<' '<<mem<<' '<<dp<<' '<<ct<<' '<<mt<<'\n';
  // Establish a real reset assertion before the first clock edge. Starting
  // high before the model's initial eval does not create an asynchronous
  // edge in a two-state event simulator.
  simdEval(d);d.io_coreReset=1;d.io_memoryReset=1;simdEval(d);
  for(unsigned i=0;i<64;++i)step();d.io_coreReset=0;d.io_memoryReset=0;
  auto releasedMem=memoryEdges,releasedCore=coreEdges;
  while(memoryEdges<releasedMem+4||coreEdges<releasedCore+4)step();start=now;
 }
 unsigned random(){rng=rng*1664525u+1013904223u;return rng;}
 void step(){
  auto t=std::min(ct,mt);if(dram)t=std::min(t,dt);now=t;
  bool mr=t==mt&&!d.io_memoryClock;
  if(mr)++memoryEdges;if(t==ct&&!d.io_coreClock)++coreEdges;
  if(dram&&t==dt){dram->ClockTick();dt+=dp;}
  if(!reading&&!reads.empty()){
   readAddress=reads.front();reads.pop_front();readMask=0;reading=true;readBeat=random()%4;
  }
  simdEval(d);
  d.io_memoryRequest_ready=(rng&7)!=0&&(!dram||dram->WillAcceptTransaction(d.io_memoryRequest_bits_address,false));
  d.io_writeRequest_ready=!writing&&(rng&15)!=0&&(!dram||dram->WillAcceptTransaction(d.io_writeRequest_bits_address,true));
  d.io_writeData_ready=writing&&(rng&3)!=0;
  d.io_memoryResponse_valid=reading;
  d.io_memoryResponse_bits_id=0;d.io_memoryResponse_bits_error=reading&&readFaults.count(readAddress);
  d.io_memoryResponse_bits_beat=readBeat;
  for(unsigned i=0;i<4;++i){
   uint32_t value=0;
   if(reading)for(unsigned b=0;b<4;++b)value|=uint32_t(memory[readAddress+16*readBeat+4*i+b])<<(8*b);
   d.io_memoryResponse_bits_data[i]=value;
  }
  d.io_writeResponse_valid=!writes.empty();d.io_writeResponse_bits_id=0;
  d.io_writeResponse_bits_error=!writes.empty()&&writeFaults.count(writes.front());
  simdEval(d);commandAccepted=mr&&d.io_command_valid&&d.io_command_ready;
  if(commandAccepted&&!timerStarted){
   start=now;timerStarted=true;std::cout<<"T "<<d.io_command_bits_tag<<' '<<start<<'\n';
  }
  if(commandAccepted){prefetchLoadBytes=0;prefetchCoreBytes=0;}
#if HAS_NATIVE_PREFETCH
  if(mr&&d.io_prefetchLoad_valid){
   prefetchLoadBytes+=d.io_prefetchLoad_bits;
   if(d.io_coreActive)prefetchCoreBytes+=d.io_prefetchLoad_bits;
  }
#endif
  bool rr=mr&&d.io_memoryResponse_valid&&d.io_memoryResponse_ready;
  bool wr=mr&&d.io_writeResponse_valid&&d.io_writeResponse_ready;
  if(mr&&!d.io_memoryReset){
   if(d.io_memoryRequest_valid&&d.io_memoryRequest_ready){
    uint64_t a=d.io_memoryRequest_bits_address;
    if(d.io_memoryRequest_bits_id!=0||a%64)throw std::runtime_error("gather request protocol");
    if(dram){if(!dram->AddTransaction(a,false))throw std::runtime_error("DRAM rejected read handshake");}
    else reads.push_back(a);++readRequests;
    maxReadInFlight=std::max(maxReadInFlight,++readInFlight);
    if(readInFlight>1)throw std::runtime_error("single-line GEMM gather exceeded one outstanding read");
   }
   if(d.io_writeRequest_valid&&d.io_writeRequest_ready){
    if(writing||d.io_writeRequest_bits_id!=0)throw std::runtime_error("scatter request protocol");
    writeAddress=d.io_writeRequest_bits_address;writing=true;writeBeat=0;++writeRequests;
    maxWriteInFlight=std::max(maxWriteInFlight,++writeInFlight);
    if(writeInFlight>1)throw std::runtime_error("write DMA exceeded one outstanding transaction");
   }
   if(d.io_writeData_valid&&d.io_writeData_ready){
    ++writeDataBeats;
    if(!writing||d.io_writeData_bits_beat!=writeBeat||d.io_writeData_bits_id!=0)throw std::runtime_error("scatter beat protocol");
    for(unsigned b=0;b<16;++b)if(d.io_writeData_bits_enable>>b&1){
     uint8_t v=d.io_writeData_bits_data[b/4]>>(8*(b%4));auto a=writeAddress+16*writeBeat+b;
     memory[a]=v;written[a]=v;++writeBytes;
    }
    if(++writeBeat==4){
     if(dram){if(!dram->AddTransaction(writeAddress,true))throw std::runtime_error("DRAM rejected write data");}
     else writes.push_back(writeAddress);writing=false;
    }
   }
   if(d.io_done_valid&&d.io_done_ready){
    if(bool(d.io_done_bits_error)!=expectedFailure)throw std::runtime_error("GEMM error status disagrees with injected memory fault");
    if(expectedFailure)std::cout<<"F "<<d.io_done_bits_tag<<" 1\n";
    expectedFailure=false;
    ++commandCount;submitted=now;
    std::cout<<"D "<<d.io_done_bits_tag<<' '<<unsigned(d.io_done_bits_flags)<<' '
     <<d.io_done_bits_computeCycles<<' '<<d.io_done_bits_macs<<' '<<d.io_done_bits_overflow<<' '
     <<d.io_done_bits_memoryCycles<<' '<<d.io_done_bits_readBursts<<' '<<now<<' '
     <<d.io_done_bits_vectors<<' '<<d.io_done_bits_feedbackStalls<<' '<<d.io_done_bits_inputStalls<<' '
     <<d.io_done_bits_dequantOperations<<' '<<d.io_done_bits_dequantCycles<<std::endl;
#if HAS_NATIVE_PREFETCH
    if(prefetchLoadBytes!=d.io_done_bits_prefetchLoadBytes)throw std::runtime_error("prefetch load byte counter mismatch");
    std::cout<<"N "<<d.io_done_bits_tag<<' '<<d.io_done_bits_prefetchedTiles<<' '
     <<d.io_done_bits_prefetchLoadBytes<<' '<<d.io_done_bits_prefetchInFlightBytes<<' '<<prefetchCoreBytes<<std::endl;
#endif
#if HAS_PHASE_COUNTERS
    std::cout<<"Z "<<d.io_done_bits_tag<<' '<<d.io_done_bits_stageCycles_0<<' '
     <<d.io_done_bits_stageCycles_1<<' '<<d.io_done_bits_stageCycles_2<<' '<<d.io_done_bits_stageCycles_3<<' '
     <<d.io_done_bits_stageCycles_4<<' '<<d.io_done_bits_stageCycles_5<<' '<<d.io_done_bits_stageCycles_6<<' '
     <<d.io_done_bits_stageCycles_7<<' '<<d.io_done_bits_stageCycles_8<<std::endl;
#endif
#if HAS_OUTPUT_EPILOGUE
    std::cout<<"Y "<<d.io_done_bits_tag<<' '<<d.io_done_bits_epilogueElements<<' '
     <<d.io_done_bits_epilogueAuxVectors<<' '<<d.io_done_bits_epilogueCycles<<' '<<d.io_done_bits_epilogueSaturations<<std::endl;
#endif
   }
  }
  if(t==ct){d.io_coreClock=!d.io_coreClock;ct+=ch;}
  if(t==mt){d.io_memoryClock=!d.io_memoryClock;mt+=mh;if(!d.io_memoryClock){random();d.io_done_ready=(rng&3)!=0;}}
  simdEval(d);
  if(rr){
   ++readDataBeats;
   readMask|=1u<<readBeat;
   if(readMask==15){readFaults.erase(readAddress);reading=false;
    if(!readInFlight)throw std::runtime_error("read accounting underflow");--readInFlight;}
   else {do{readBeat=(readBeat+1)%4;}while(readMask>>readBeat&1);}
  }
  if(wr){writeFaults.erase(writes.front());writes.pop_front();
   if(!writeInFlight)throw std::runtime_error("write accounting underflow");--writeInFlight;}
  if(now-start>1000000000000ULL)throw std::runtime_error("GEMM simulation timeout");
 }
 void command(const Descriptor& x,bool wait=true){
  auto expected=commandCount+1;
  descriptorReadStart=readRequests;descriptorWriteStart=writeBytes;
  d.io_command_bits_a=x.a;d.io_command_bits_b=x.b;d.io_command_bits_c=x.c;
  d.io_command_bits_aStride=x.as;d.io_command_bits_bStride=x.bs;d.io_command_bits_cStride=x.cs;
  d.io_command_bits_m=x.m;d.io_command_bits_n=x.n;d.io_command_bits_k=x.k;
  d.io_command_bits_fmt=x.fmt;d.io_command_bits_tag=x.tag;d.io_command_valid=1;
  d.io_command_bits_weight4=x.weight4;
  d.io_command_bits_accumulate=x.accumulate;
  d.io_command_bits_weightScales=x.scales;d.io_command_bits_scaleStride=x.scaleStride;d.io_command_bits_groupLog2=x.groupLog2;
  d.io_command_bits_grouped=x.grouped;d.io_command_bits_activationScales=x.activationScales;
  d.io_command_bits_activationScaleStride=x.activationScaleStride;
#if HAS_NATIVE_PREFETCH
  d.io_command_bits_prefetch=x.prefetch;
#endif
#if HAS_OUTPUT_EPILOGUE
  d.io_command_bits_outputConvert=x.outputConvert;d.io_command_bits_outputFormat=x.outputFormat;
  d.io_command_bits_outputScale=x.outputScale;d.io_command_bits_outputZeroPoint=x.outputZeroPoint;
#endif
  do{step();}while(!commandAccepted);d.io_command_valid=0;
  if(wait)while(commandCount<expected)step();
 }
#if HAS_SCRATCHPAD
 void scratch(unsigned bank,unsigned address,bool write,unsigned strobe,const std::string& value,unsigned tag){
  d.io_scratch_bits_bank=bank;d.io_scratch_bits_address=address;d.io_scratch_bits_write=write;
  d.io_scratch_bits_strobe=strobe;d.io_scratch_bits_tag=tag;setWide(d.io_scratch_bits_data,value,4);
  d.io_scratch_valid=1;d.io_scratchResult_ready=0;
  auto begin=memoryEdges;
  for(;;){
   if(memoryEdges>begin+100)throw std::runtime_error("scratch request timeout");
   simdEval(d);
   // step() chooses the next event; sample the handshake at that event.
   auto t=std::min(ct,mt);if(dram)t=std::min(t,dt);
   bool accepted=t==mt&&!d.io_memoryClock&&d.io_scratch_ready;
   step();if(accepted)break;
  }
  d.io_scratch_valid=0;
  while(!d.io_scratchResult_valid){step();if(memoryEdges>begin+100)throw std::runtime_error("scratch response timeout");}
  auto held=wideHex(d.io_scratchResult_bits_data,4);auto error=d.io_scratchResult_bits_error;
  if(d.io_scratchResult_bits_tag!=tag)throw std::runtime_error("scratch response tag mismatch");
  auto edges=memoryEdges;
  while(memoryEdges<edges+7){step();if(!d.io_scratchResult_valid||wideHex(d.io_scratchResult_bits_data,4)!=held||
    d.io_scratchResult_bits_error!=error||d.io_scratchResult_bits_tag!=tag)throw std::runtime_error("stalled scratch response changed");}
  std::cout<<"C "<<tag<<' '<<unsigned(error)<<' '<<held<<'\n';d.io_scratchResult_ready=1;
  while(d.io_scratchResult_valid)step();d.io_scratchResult_ready=0;
 }
#endif
 void resetInFlight(char phase){
  if(phase!='r'&&phase!='c'&&phase!='w')throw std::runtime_error("reset phase must be read/core/write");
  auto before=commandCount;
  while((phase=='r'&&readRequests==descriptorReadStart)||(phase=='c'&&!d.io_coreActive)||
        (phase=='w'&&writeBytes==descriptorWriteStart)){
   step();if(commandCount!=before)throw std::runtime_error("reset did not interrupt an active descriptor");
  }
  const auto* resetDomain=std::getenv("SIMD_RESET_DOMAIN");
  std::string domain=resetDomain?resetDomain:"both";
  if(domain!="both"&&domain!="core"&&domain!="memory")throw std::runtime_error("invalid reset domain");
  d.io_command_valid=0;d.io_coreReset=domain!="memory";d.io_memoryReset=domain!="core";simdEval(d);
  std::cout<<"XR "<<domain<<' '<<unsigned(d.io_coreReset)<<' '<<unsigned(d.io_memoryReset)<<'\n';
  // Both endpoints enter reset. Cancel a partial front-end write burst and
  // drain transactions already accepted by DRAM before reset is released.
  writing=false;
  if(dram){
   dram->RequestDrain();while(!dram->IsDrained())step();
   oldReadBursts+=dram->GetReadBursts();oldWriteBursts+=dram->GetWriteBursts();dram->PrintStats();
   dram.reset();
  }
  reads.clear();writes.clear();reading=false;readFaults.clear();writeFaults.clear();expectedFailure=false;
  readInFlight=0;writeInFlight=0;
  d.io_memoryResponse_valid=0;d.io_writeResponse_valid=0;
  epochWriteRequests=writeRequests;epochWriteBytes=writeBytes;
  if(memoryConfiguration!="-"){
   auto directory=memoryOutput+"/epoch_"+std::to_string(++epoch);
   std::filesystem::create_directories(directory);
   dram.reset(new SimdDramAdapter(memoryConfiguration,directory,
    [this](uint64_t a){reads.push_back(a);},[this](uint64_t a){writes.push_back(a);}));
  }else ++epoch;
  auto heldCore=coreEdges,heldMemory=memoryEdges;
  while(coreEdges<heldCore+4||memoryEdges<heldMemory+4)step();
  d.io_coreReset=0;d.io_memoryReset=0;simdEval(d);
  auto releasedCore=coreEdges,releasedMemory=memoryEdges;
  while(coreEdges<releasedCore+4||memoryEdges<releasedMemory+4)step();
  std::cout<<"X "<<epoch<<' '<<phase<<' '<<now<<'\n';
 }
 void finish(){
  if(dram){dram->RequestDrain();while(!dram->IsDrained())step();dram->PrintStats();}
  if(reading||writing||!reads.empty()||!writes.empty())throw std::runtime_error("unconsumed memory transactions");
  if(readInFlight||writeInFlight)throw std::runtime_error("outstanding external transactions after final drain");
  if(!epoch&&(readDataBeats!=4*readRequests||writeDataBeats!=4*writeRequests))
   throw std::runtime_error("128-bit data beat accounting differs from 64-byte requests");
  std::cout<<"PORTS "<<maxReadInFlight<<' '<<maxWriteInFlight<<' '<<readDataBeats<<' '<<writeDataBeats<<'\n';
  if(d.io_writeBytes!=writeBytes-epochWriteBytes||d.io_writeBursts!=writeRequests-epochWriteRequests)throw std::runtime_error("scatter counters disagree");
  for(auto [a,b]:written)std::cout<<"W "<<a<<' '<<unsigned(b)<<'\n';
  std::cout<<"H "<<d.io_sramReadBytes<<' '<<d.io_sramWriteBytes<<' '
   <<d.io_aggregateReadBytes<<' '<<d.io_aggregateWriteBytes<<'\n';
  std::cout<<"S "<<start<<' '<<submitted<<' '<<now<<' '<<readRequests<<' '<<writeRequests<<' '<<writeBytes<<' '
   <<(oldReadBursts+(dram?dram->GetReadBursts():0))<<' '<<(oldWriteBursts+(dram?dram->GetWriteBursts():0))<<'\n';
  simdWriteCoverage();d.final();
 }
};
int main(int argc,char**argv){
 try{
  if(argc!=7){std::cerr<<"usage: gemm_sim trace core_ps memory_ps dram_ini_or_- dram_out seed\n";return 2;}
  Verilated::commandArgs(argc,argv);simdConfigureAssertions();Verilated::threadContextp()->threads(SIMULATION_THREADS);
  Simulation s(std::stoull(argv[2]),std::stoull(argv[3]),argv[4],argv[5],std::stoul(argv[6]));
  SimdCoverageSave coverage;
  std::ifstream f(argv[1]);if(!f)throw std::runtime_error("trace not found");std::string line;
  while(std::getline(f,line)){
   std::istringstream in(line);char kind=0;in>>kind;if(!kind||kind=='#')continue;
   if(kind=='L'){
    uint64_t a;std::string hex;in>>a>>hex;if(hex.size()!=32)throw std::runtime_error("128-bit source line required");
    for(unsigned i=0;i<16;++i)s.memory[a+i]=std::stoul(hex.substr(30-2*i,2),nullptr,16);
   }else if(kind=='F'){
    char direction=0;uint64_t address=0;in>>direction>>address;
    if(!in||address%64||(direction!='r'&&direction!='w')||s.expectedFailure)throw std::runtime_error("invalid memory fault injection");
    (direction=='r'?s.readFaults:s.writeFaults).insert(address);s.expectedFailure=true;
   }else if(kind=='X'){
    char phase=0;in>>phase;if(!in)throw std::runtime_error("missing reset phase");s.resetInFlight(phase);
#if HAS_SCRATCHPAD
   }else if(kind=='C'){
    unsigned bank,address,write,tag,strobe;std::string data;
    in>>bank>>address>>write>>std::hex>>strobe>>data>>std::dec>>tag;
    if(!in||bank>3||address>8191||write>1||strobe>65535||data.size()>32)throw std::runtime_error("invalid scratch command");
    s.scratch(bank,address,write,strobe,data,tag);
#endif
   }else if(kind=='O'){
    unsigned enabled=2,format=0;uint32_t scale=0;int64_t zero=0;
    in>>enabled>>format>>std::hex>>scale>>std::dec>>zero;
    if(!in||enabled>1||format>10||zero<INT32_MIN||zero>INT32_MAX||!HAS_OUTPUT_EPILOGUE)
     throw std::runtime_error("invalid or unsupported output conversion control");
    s.outputConvert=enabled;s.outputFormat=format;s.outputScale=scale;s.outputZeroPoint=int32_t(zero);
   }else if(kind=='V'){
    unsigned enabled=2;in>>enabled;
    if(!in||enabled>1||!HAS_NATIVE_PREFETCH)throw std::runtime_error("invalid or unsupported autonomous prefetch control");
    s.allowPrefetch=enabled;
   }else if(kind=='G'||kind=='B'){
    Descriptor x;x.prefetch=s.allowPrefetch;
    x.outputConvert=s.outputConvert;x.outputFormat=s.outputFormat;x.outputScale=s.outputScale;x.outputZeroPoint=s.outputZeroPoint;
    in>>x.fmt>>x.m>>x.n>>x.k>>x.a>>x.b>>x.c>>x.as>>x.bs>>x.cs>>x.tag;
    if(!in)throw std::runtime_error("invalid GEMM descriptor");
    unsigned mixed=0,initial=0;if(in>>mixed){
     if(mixed>1)throw std::runtime_error("invalid weight packing");x.weight4=mixed;
     if(in>>initial){
      if(initial>1)throw std::runtime_error("invalid C accumulation mode");x.accumulate=initial;
      if(in>>x.scales){
       if(!(in>>x.scaleStride>>x.groupLog2))throw std::runtime_error("invalid group scale descriptor");
       unsigned grouped=0;if(in>>grouped){
        if(grouped>1||!(in>>x.activationScales>>x.activationScaleStride))throw std::runtime_error("invalid activation scale descriptor");
        x.grouped=grouped;
       }
      }
     }
    }
    s.command(x,kind=='G');
   }else throw std::runtime_error("unknown trace command");
  }
  s.finish();return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
