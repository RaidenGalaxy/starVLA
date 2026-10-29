#include "VTileSystem.h"
#include "verilated.h"
#include "coverage.hpp"
#include "ports.hpp"
#include "wide.hpp"
#include "assertions.hpp"
#include <cstdint>
#include <cmath>
#include "dram_adapter.hpp"
#include "clocking.hpp"
#ifndef SIMULATION_THREADS
#define SIMULATION_THREADS 1
#endif
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

struct Beat {unsigned bank,address;std::string data;};
struct ReadBurst {uint64_t address;std::vector<Beat> beats;bool ready=false;};
struct NativeBurst {uint64_t address;unsigned id;};
class Simulation {
 public:
 VTileSystem d;
 uint64_t now=0,coreHalf,memHalf,ct=0,mt=0,dt=0,dramPeriod=0;
 uint64_t memoryEdges=0,doneCount=0,resultCount=0,startTime=0,submittedTime=0;
 uint64_t coreEdges=0;
 uint64_t lastProgress=0;
 uint64_t coreCompute=0,macs=0,loadBytes=0,writePayload=0,dmaSerial=0;
 bool loadAccepted=false,commandAccepted=false,readAccepted=false,auxAccepted=false;
 bool randomized=false;uint32_t rng=207;
 std::unique_ptr<SimdDramAdapter> dram;
 std::map<uint64_t,std::deque<ReadBurst*>> waiting;
 std::deque<ReadBurst*> returned;
 std::map<unsigned,std::string> values;
 std::string lastValue;std::vector<uint8_t> outputBuffer;
 std::map<uint64_t,uint8_t> shadowWrites;
 std::vector<uint8_t> expectedWrites;
 uint64_t auxiliaryCount=0,auxiliaryPs=0;
 uint64_t overflowEvents=0,exceptionOps=0;
 uint64_t vectors=0,feedbackStalls=0,inputStalls=0;
 uint64_t observedReadBursts=0,observedWriteBursts=0;
 uint64_t overlappingLoadBytes=0,overlappingLoadEdges=0;
 bool nativeDma=true;
 std::map<uint64_t,std::string> nativeMemory;
 std::map<uint64_t,std::deque<NativeBurst>> nativeWaiting;
 std::deque<NativeBurst> nativeReturned;
 NativeBurst nativeCurrent{};bool nativeActive=false;unsigned nativeBeat=0;
 bool dmaCommandAccepted=false;uint64_t dmaDoneCount=0;
 bool outputCommandAccepted=false,outputPushAccepted=false,nativeWriteHeader=false;
 uint64_t nativeWriteAddress=0,nativeOutputBytes=0,nativeOutputDone=0;
 unsigned nativeWriteBeat=0,nativeLastTag=0;
 std::deque<uint64_t> nativeWriteReturned;
 Simulation(uint64_t core,uint64_t mem,const std::string& config,const std::string& out,unsigned seed,bool useNative=true)
  :coreHalf(core/2),memHalf(mem/2){
  nativeDma=useNative;
  randomized=seed!=0;rng=seed;
  if(!coreHalf||!memHalf||core%2||mem%2)throw std::runtime_error("clock periods must be even positive picoseconds");
  ct=simdClockPhase("SIMD_CORE_PHASE_PS",core);mt=simdClockPhase("SIMD_MEMORY_PHASE_PS",mem);dt=ct;
  if(d.threads()!=SIMULATION_THREADS)throw std::runtime_error("generated root thread count differs from simulation configuration");
  d.io_coreClock=0;d.io_memoryClock=0;d.io_coreReset=0;d.io_memoryReset=0;
  d.io_load_valid=0;d.io_command_valid=0;d.io_read_valid=0;d.io_auxiliary_valid=0;
  d.io_seed_valid=0;d.io_dmaCommand_valid=0;d.io_dmaDone_ready=1;
  d.io_memoryRequest_ready=0;d.io_memoryResponse_valid=0;
  d.io_outputStart_valid=0;d.io_outputPush_valid=0;d.io_outputDone_ready=1;
#if HAS_OUTPUT_PUSH_OVERRIDE
  d.io_outputPush_bits_overrideData=0;setWide(d.io_outputPush_bits_data,"0");
#endif
  d.io_writeRequest_ready=0;d.io_writeData_ready=0;d.io_writeResponse_valid=0;
  d.io_result_ready=1;d.io_done_ready=1;
  d.io_peek_valid=0;d.io_peek_bits_bank=0;d.io_peek_bits_address=0;d.io_peekResult_ready=1;
#if HAS_AGGREGATE_INSPECTION
  d.io_aggregateInspection_valid=0;
  d.io_aggregateInspectionResult_ready=1;
#endif
  // Explicit 0->1 assertion initializes asynchronous reset synchronizers
  // before any core edge, including in two-state hierarchical simulation.
  simdEval(d);d.io_coreReset=1;d.io_memoryReset=1;simdEval(d);
  if(config!="-") {
   std::filesystem::create_directories(out);
   dram.reset(new SimdDramAdapter(config,out,[&](uint64_t address){
    if(nativeDma) {
     auto &q=nativeWaiting.at(address);nativeReturned.push_back(q.front());q.pop_front();
    }else{
     auto &q=waiting.at(address);auto *x=q.front();q.pop_front();x->ready=true;returned.push_back(x);
    }
   },[&](uint64_t address){if(nativeDma)nativeWriteReturned.push_back(address);}));
   dramPeriod=uint64_t(std::llround(dram->GetTCK()*1000.0));
   if(!dramPeriod)throw std::runtime_error("DRAM tCK must be positive");
  }
  std::cout<<"J "<<core<<' '<<mem<<' '<<dramPeriod<<' '<<ct<<' '<<mt<<'\n';
  while(memoryEdges<8)step();d.io_coreReset=0;d.io_memoryReset=0;
  auto releasedMem=memoryEdges,releasedCore=coreEdges;
  while(memoryEdges<releasedMem+4||coreEdges<releasedCore+4)step();startTime=now;
 }
 ~Simulation(){d.final();}
 void step(){
  uint64_t t=std::min(ct,mt);if(dram)t=std::min(t,dt);
  now=t;const bool cr=t==ct&&!d.io_coreClock,mr=t==mt&&!d.io_memoryClock;
  if(cr)++coreEdges;
  // DRAM advances before a memory-port handshake at the same timestamp.
  if(dram&&t==dt){dram->ClockTick();dt+=dramPeriod;}
  if(nativeDma) {
   simdEval(d);
   d.io_memoryRequest_ready=!dram || dram->WillAcceptTransaction(d.io_memoryRequest_bits_address,false);
   d.io_writeRequest_ready=!nativeWriteHeader && (!dram || dram->WillAcceptTransaction(d.io_writeRequest_bits_address,true));
   d.io_writeData_ready=nativeWriteHeader;
   d.io_writeResponse_valid=!nativeWriteReturned.empty();
   d.io_writeResponse_bits_id=0;d.io_writeResponse_bits_error=0;
   if(!nativeActive&&!nativeReturned.empty()) {
    nativeCurrent=nativeReturned.front();nativeReturned.pop_front();nativeBeat=0;nativeActive=true;
   }
   d.io_memoryResponse_valid=nativeActive;
   if(nativeActive) {
    d.io_memoryResponse_bits_id=nativeCurrent.id;d.io_memoryResponse_bits_beat=nativeBeat;
    d.io_memoryResponse_bits_error=0;
    auto it=nativeMemory.find(nativeCurrent.address+16*nativeBeat);
    setWide(d.io_memoryResponse_bits_data,it==nativeMemory.end()?"0":it->second,4);
   }
  }
  simdEval(d);loadAccepted=mr&&d.io_load_valid&&d.io_load_ready;
  const uint64_t deliveredBefore=d.io_dmaPayloadBytes;
  const bool activeBefore=d.io_coreActive;
  commandAccepted=mr&&d.io_command_valid&&d.io_command_ready;
  readAccepted=mr&&d.io_read_valid&&d.io_read_ready;
  auxAccepted=mr&&d.io_auxiliary_valid&&d.io_auxiliary_ready;
  dmaCommandAccepted=mr&&d.io_dmaCommand_valid&&d.io_dmaCommand_ready;
  outputCommandAccepted=mr&&d.io_outputStart_valid&&d.io_outputStart_ready;
  outputPushAccepted=mr&&d.io_outputPush_valid&&d.io_outputPush_ready;
  bool memoryResponseAccepted=mr&&d.io_memoryResponse_valid&&d.io_memoryResponse_ready;
  bool writeResponseAccepted=mr&&d.io_writeResponse_valid&&d.io_writeResponse_ready;
  if(loadAccepted||commandAccepted||readAccepted||auxAccepted||dmaCommandAccepted||outputCommandAccepted||
     outputPushAccepted||memoryResponseAccepted||writeResponseAccepted||
     (mr&&((d.io_result_valid&&d.io_result_ready)||(d.io_done_valid&&d.io_done_ready)||
       (d.io_dmaDone_valid&&d.io_dmaDone_ready)||(d.io_outputDone_valid&&d.io_outputDone_ready)||
       (d.io_memoryRequest_valid&&d.io_memoryRequest_ready)||(d.io_writeRequest_valid&&d.io_writeRequest_ready)||
       (d.io_writeData_valid&&d.io_writeData_ready))))lastProgress=now;
  if(mr){
   memoryEdges++;
   if(!d.io_memoryReset&&d.io_result_valid&&d.io_result_ready){
    values[d.io_result_bits_tag]=wideHex(d.io_result_bits_data);
    lastValue=values[d.io_result_bits_tag];
    nativeLastTag=d.io_result_bits_tag;
    if(d.io_result_bits_illegal)throw std::runtime_error("illegal RTL operation");
    std::cout<<"R "<<d.io_result_bits_tag<<' '<<values[d.io_result_bits_tag]<<' '
     <<unsigned(d.io_result_bits_flags)<<' '<<now<<'\n';resultCount++;
   }
   if(!d.io_memoryReset&&d.io_done_valid&&d.io_done_ready){
    coreCompute+=d.io_done_bits_cycles;macs+=d.io_done_bits_macs;doneCount++;
    overflowEvents+=d.io_done_bits_overflow;exceptionOps+=d.io_done_bits_exceptions;
    vectors+=d.io_done_bits_vectors;feedbackStalls+=d.io_done_bits_feedbackStalls;inputStalls+=d.io_done_bits_inputStalls;
    std::cout<<"D "<<d.io_done_bits_tag<<' '<<d.io_done_bits_cycles<<' '<<d.io_done_bits_macs<<' '<<now<<'\n';
   }
   if(!d.io_memoryReset&&d.io_dmaDone_valid&&d.io_dmaDone_ready) {
    if(d.io_dmaDone_bits_error)throw std::runtime_error("RTL DMA reported a memory error");
    ++dmaDoneCount;
   }
   if(!d.io_memoryReset&&d.io_outputDone_valid&&d.io_outputDone_ready) {
    if(d.io_outputDone_bits_error)throw std::runtime_error("RTL output DMA reported a memory error");
    ++nativeOutputDone;
   }
   if(!d.io_memoryReset&&d.io_writeRequest_valid&&d.io_writeRequest_ready) {
    ++observedWriteBursts;
    if(nativeWriteHeader||d.io_writeRequest_bits_id!=0)throw std::runtime_error("invalid output DMA header");
    nativeWriteHeader=true;nativeWriteAddress=d.io_writeRequest_bits_address;nativeWriteBeat=0;
   }
   if(!d.io_memoryReset&&d.io_writeData_valid&&d.io_writeData_ready) {
    if(!nativeWriteHeader||d.io_writeData_bits_beat!=nativeWriteBeat||d.io_writeData_bits_id!=0)
     throw std::runtime_error("invalid output DMA beat");
    for(unsigned i=0;i<16;++i)if(d.io_writeData_bits_enable>>i&1)
     shadowWrites[nativeWriteAddress+16*nativeWriteBeat+i]=d.io_writeData_bits_data[i/4]>>(8*(i%4));
    if(++nativeWriteBeat==4) {
     if(dram) {
      if(!dram->AddTransaction(nativeWriteAddress,true))throw std::runtime_error("DRAM rejected output data");
     }else nativeWriteReturned.push_back(nativeWriteAddress);
     nativeWriteHeader=false;
    }
   }
   if(!d.io_memoryReset&&d.io_memoryRequest_valid&&d.io_memoryRequest_ready) {
    ++observedReadBursts;
    NativeBurst burst{d.io_memoryRequest_bits_address,unsigned(d.io_memoryRequest_bits_id)};
    if(!nativeDma)throw std::runtime_error("unexpected native DMA request");
    if(dram) {
     nativeWaiting[burst.address].push_back(burst);
     if(!dram->AddTransaction(burst.address,false))throw std::runtime_error("DRAM rejected an accepted RTL request");
    }else nativeReturned.push_back(burst);
   }
  }
  if(t==ct){d.io_coreClock=!d.io_coreClock;ct+=coreHalf;}
  if(t==mt){
   d.io_memoryClock=!d.io_memoryClock;mt+=memHalf;
   if(!d.io_memoryClock&&randomized){
    rng=rng*1664525u+1013904223u;d.io_result_ready=(rng&3)!=0;
    rng=rng*1664525u+1013904223u;d.io_done_ready=(rng&3)!=0;
   }
  }
  simdEval(d);
  if(mr&&!d.io_memoryReset&&activeBefore&&d.io_dmaPayloadBytes>deliveredBefore){
   overlappingLoadBytes+=d.io_dmaPayloadBytes-deliveredBefore;++overlappingLoadEdges;
  }
  if(memoryResponseAccepted) {
   if(++nativeBeat==4)nativeActive=false;
  }
  if(writeResponseAccepted)nativeWriteReturned.pop_front();
  if(now-startTime>1000000000000ULL)throw std::runtime_error("simulation timeout");
  if(now-lastProgress>100000*std::max({coreHalf,memHalf,dramPeriod}))
   throw std::runtime_error("system made no interface progress for 100000 clock half-periods");
 }
 void loadDirect(const std::vector<Beat>& beats){
  if(beats.empty())return;
  std::vector<ReadBurst> bursts;
  for(std::size_t i=0;i<beats.size();i+=4){
   ReadBurst b;b.address=0x10000000ULL+(dmaSerial++*64);
   b.beats.assign(beats.begin()+i,beats.begin()+std::min(beats.size(),i+4));bursts.push_back(b);
  }
  std::size_t issued=0,finished=0,index=0;ReadBurst* current=nullptr;
  uint64_t lastIssueEdge=~0ULL;d.io_load_valid=0;
  while(finished<bursts.size()){
   if(issued<bursts.size()&&issued-finished<32&&lastIssueEdge!=memoryEdges){
    auto &b=bursts[issued];
    if(!dram || (dram->WillAcceptTransaction(b.address,false)&&dram->AddTransaction(b.address,false))){
     if(dram)waiting[b.address].push_back(&b);else{b.ready=true;returned.push_back(&b);}
     issued++;lastIssueEdge=memoryEdges;
    }
   }
   if(!current&&!returned.empty()){current=returned.front();returned.pop_front();index=0;}
   if(current){
    auto &b=current->beats[index];d.io_load_valid=1;d.io_load_bits_bank=b.bank;
    d.io_load_bits_address=b.address;d.io_load_bits_strobe=0xffff;setWide(d.io_load_bits_data,b.data,4);
   }else d.io_load_valid=0;
   step();
   if(loadAccepted){
    loadBytes+=16;index++;
    if(index==current->beats.size()){current=nullptr;finished++;d.io_load_valid=0;}
   }
  }
  d.io_load_valid=0;
 }
 void load(const std::vector<Beat>& beats) {
  if(!nativeDma){loadDirect(beats);return;}
  for(std::size_t begin=0;begin<beats.size();) {
   std::size_t end=begin+1;
   while(end<beats.size()&&beats[end].bank==beats[begin].bank&&beats[end].address==beats[begin].address+(end-begin))++end;
   unsigned count=end-begin;uint64_t source=0x10000000ULL+dmaSerial*64;
   dmaSerial+=(count+3)/4;
   for(unsigned i=0;i<count;++i)nativeMemory[source+16*i]=beats[begin+i].data;
   auto previous=dmaDoneCount;auto before=d.io_dmaPayloadBytes;
   d.io_dmaCommand_bits_source=source;d.io_dmaCommand_bits_bank=beats[begin].bank;
   d.io_dmaCommand_bits_destination=beats[begin].address;d.io_dmaCommand_bits_beats=count;
   d.io_dmaCommand_bits_tag=previous;d.io_dmaCommand_valid=1;
   do{step();}while(!dmaCommandAccepted);d.io_dmaCommand_valid=0;
   while(dmaDoneCount==previous)step();
   auto delivered=d.io_dmaPayloadBytes-before;
   if(delivered!=16*count)throw std::runtime_error("DMA local delivery count mismatch");
   loadBytes+=delivered;begin=end;
  }
 }
 void command(unsigned fmt,unsigned contexts,unsigned k,unsigned clear,unsigned rowmask,
              const std::vector<unsigned>& masks,unsigned tag){
  d.io_command_bits_fmt=fmt;d.io_command_bits_contexts=contexts;d.io_command_bits_k=k;
  d.io_command_bits_clear=clear;d.io_command_bits_rowMask=rowmask;d.io_command_bits_tag=tag;
  for(int i=0;i<COLS;i++)setCommandMask(d,i,masks.at(i));
  d.io_command_valid=1;do{step();}while(!commandAccepted);d.io_command_valid=0;
 }
 void waitDone(uint64_t target){while(doneCount<target)step();}
 void read(unsigned ctx,unsigned row,unsigned col,unsigned tag){
  d.io_read_bits_ctx=ctx;d.io_read_bits_row=row;d.io_read_bits_col=col;d.io_read_bits_tag=tag;
  d.io_read_valid=1;do{step();}while(!readAccepted);d.io_read_valid=0;
  while(!values.count(tag))step();
 }
 void configureOutput(uint64_t bytes) {
  nativeOutputBytes=bytes;if(!nativeDma||!bytes)return;
  d.io_outputStart_bits_address=0x80000000ULL;d.io_outputStart_bits_bytes=bytes;
  d.io_outputStart_bits_tag=0;d.io_outputStart_valid=1;
  do{step();}while(!outputCommandAccepted);d.io_outputStart_valid=0;
 }
 void writeback(uint64_t bytes){
  writePayload+=bytes;
  if(bytes>64||lastValue.empty())throw std::runtime_error("invalid result writeback");
  if(nativeDma) {
   // This is an independent consumer of retired RTL data for validation;
   // the write DMA receives its payload from the device's own result register.
   for(unsigned i=0;i<bytes;++i)expectedWrites.push_back(std::stoul(lastValue.substr(lastValue.size()-2*(i+1),2),nullptr,16));
   d.io_outputPush_bits_bytes=bytes;d.io_outputPush_bits_resultTag=nativeLastTag;d.io_outputPush_valid=1;
   do{step();}while(!outputPushAccepted);d.io_outputPush_valid=0;return;
  }
  for(unsigned i=0;i<bytes;i++)outputBuffer.push_back(std::stoul(lastValue.substr(lastValue.size()-2*(i+1),2),nullptr,16));
  flushOutput(false);
 }
 void flushOutput(bool tail){
  while(outputBuffer.size()>=64||(tail&&!outputBuffer.empty())){
   auto bytes=std::min<std::size_t>(64,outputBuffer.size());
   uint64_t address=0x80000000ULL+(dmaSerial++*64);
   if(dram){while(!dram->WillAcceptTransaction(address,true))step();dram->AddTransaction(address,true);}
   for(unsigned i=0;i<bytes;i++)shadowWrites[address+i]=outputBuffer[i];
   outputBuffer.erase(outputBuffer.begin(),outputBuffer.begin()+bytes);
   auto until=memoryEdges+(bytes+15)/16;
   while(memoryEdges<until)step();
  }
  submittedTime=now;
 }
 std::string resolve(const std::string& token){
  if(token.empty()||token[0]!='@')return token;
  auto a=token.find(':'),b=a==std::string::npos?a:token.find(':',a+1);
  unsigned tag=std::stoul(token.substr(1,a==std::string::npos?a:a-1));
  auto text=values.at(tag);if(a==std::string::npos)return text;
  unsigned shift=std::stoul(token.substr(a+1,b-a-1)),width=std::stoul(token.substr(b+1));
  if(shift%4||width%4||shift+width>512)throw std::runtime_error("invalid result bit slice");
  return text.substr(text.size()-(shift+width)/4,width/4);
 }
 void auxiliary(unsigned fmt,unsigned op,const std::string& a,const std::string& b,
                const std::string& acc,unsigned mask,unsigned tag,unsigned aggregate,
                unsigned index,unsigned clear,unsigned scale,unsigned scaleAddress){
  auto begin=now;d.io_auxiliary_bits_request_fmt=fmt;d.io_auxiliary_bits_request_op=op;
  d.io_auxiliary_bits_request_a=std::stoull(resolve(a),nullptr,16);
  d.io_auxiliary_bits_request_b=std::stoull(resolve(b),nullptr,16);
  setWide(d.io_auxiliary_bits_request_acc,resolve(acc));d.io_auxiliary_bits_request_mask=mask;
  d.io_auxiliary_bits_request_ctx=0;d.io_auxiliary_bits_request_tag=tag;
  d.io_auxiliary_bits_aggregate=aggregate;d.io_auxiliary_bits_aggregateIndex=index;
  d.io_auxiliary_bits_clearAggregate=clear;d.io_auxiliary_bits_useScale=scale;
  d.io_auxiliary_bits_scaleAddress=scaleAddress;d.io_auxiliary_valid=1;
  do{step();}while(!auxAccepted);d.io_auxiliary_valid=0;
  while(!values.count(tag))step();auxiliaryCount++;auxiliaryPs+=now-begin;
 }
 void finish(){
  if(nativeDma) {
   while(nativeOutputBytes&&!nativeOutputDone)step();
   if(d.io_dmaWritePayloadBytes!=writePayload||nativeOutputBytes!=writePayload)
    throw std::runtime_error("RTL output DMA byte accounting mismatch");
   submittedTime=now;
  }else flushOutput(true);
  if(dram){dram->RequestDrain();while(!dram->IsDrained())step();dram->PrintStats();}
  if(nativeDma) {
   if(shadowWrites.size()!=expectedWrites.size())throw std::runtime_error("output DMA byte-enable accounting mismatch");
   for(std::size_t i=0;i<expectedWrites.size();++i)
    if(shadowWrites.at(0x80000000ULL+i)!=expectedWrites[i])throw std::runtime_error("external output differs from retired RTL data");
   if(dram&&(dram->GetReadBursts()>dram->GetSubmittedReads()||dram->GetWriteBursts()>dram->GetSubmittedWrites()))
    throw std::runtime_error("physical DRAM burst counters exceed submitted transactions");
   if(d.io_dmaReadRequests!=observedReadBursts||d.io_dmaWriteRequests!=observedWriteBursts)
    throw std::runtime_error("RTL DMA and external handshake counters disagree");
   std::cout<<"B "<<d.io_dmaReadRequests<<' '<<d.io_dmaWriteRequests<<' '<<d.io_dmaPayloadBytes<<' '
    <<d.io_dmaWritePayloadBytes<<' '<<shadowWrites.size()<<' '<<observedReadBursts<<' '<<observedWriteBursts<<'\n';
  }
  std::cout<<"H "<<d.io_sramReadBytes<<' '<<d.io_sramWriteBytes<<' '<<d.io_aggregateReadBytes<<' '<<d.io_aggregateWriteBytes<<'\n';
  std::cout<<"U "<<overlappingLoadBytes<<' '<<overlappingLoadEdges<<'\n';
  std::cout<<"V "<<vectors<<' '<<feedbackStalls<<' '<<inputStalls<<'\n';
  std::cout<<"S "<<startTime<<' '<<submittedTime<<' '<<now<<' '<<coreCompute<<' '<<macs<<' '
   <<loadBytes<<' '<<writePayload<<' '<<(dram?dram->GetReadBursts():0)<<' '
   <<(dram?dram->GetWriteBursts():0)<<' '<<auxiliaryCount<<' '<<auxiliaryPs<<' '
   <<overflowEvents<<' '<<exceptionOps<<'\n';
 }
};

int main(int argc,char**argv){
 try{
  if(argc!=7&&argc!=8){std::cerr<<"usage: system_sim trace core_ps memory_ps dram_ini_or_- dram_out seed [--direct-dma]\n";return 2;}
  if(argc==8&&std::string(argv[7])!="--direct-dma")throw std::runtime_error("unknown DMA mode");
  Verilated::commandArgs(argc,argv);simdConfigureAssertions();Verilated::threadContextp()->threads(SIMULATION_THREADS);
  Simulation s(std::stoull(argv[2]),std::stoull(argv[3]),argv[4],argv[5],std::stoul(argv[6]),argc==7);
  SimdCoverageSave coverage;
  std::ifstream f(argv[1]);if(!f)return 2;
  uint64_t totalOutput=0;std::string prescan;
  while(std::getline(f,prescan)){std::istringstream line(prescan);char c=0;if(line>>c && c=='O'){uint64_t n=0;line>>n;totalOutput+=n;}}
  f.clear();f.seekg(0);s.configureOutput(totalOutput);
  std::string line;std::vector<Beat> loads;uint64_t commands=0;
  auto flush=[&](){s.load(loads);loads.clear();};
  while(std::getline(f,line)){
   std::istringstream in(line);char cmd;in>>cmd;if(!in)continue;
   if(cmd=='L'){Beat b;in>>b.bank>>b.address>>b.data;loads.push_back(b);continue;}
   flush();
   if(cmd=='C'){
    unsigned fmt,contexts,k,clear,tag;std::string rm;in>>fmt>>contexts>>k>>clear>>rm;
    std::vector<unsigned> masks(COLS);for(auto& m:masks){std::string x;in>>x;m=std::stoul(x,nullptr,16);}in>>tag;
    s.command(fmt,contexts,k,clear,std::stoul(rm,nullptr,16),masks,tag);commands++;
   }else if(cmd=='D')s.waitDone(commands);
   else if(cmd=='R'){unsigned ctx,row,col,tag;in>>ctx>>row>>col>>tag;s.read(ctx,row,col,tag);}
   else if(cmd=='O'){uint64_t bytes;in>>bytes;s.writeback(bytes);}
   else if(cmd=='A'){
    unsigned fmt,op,tag,aggregate,index,clear,scale,scaleAddress;std::string a,b,c,mask;
    in>>fmt>>op>>a>>b>>c>>mask>>tag>>aggregate>>index>>clear>>scale>>scaleAddress;
    s.auxiliary(fmt,op,a,b,c,std::stoul(mask,nullptr,16),tag,aggregate,index,clear,scale,scaleAddress);
   }
   else if(cmd=='W'){unsigned n;in>>n;while(n--)s.step();}
  }
  flush();s.waitDone(commands);s.finish();
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
 return 0;
}
