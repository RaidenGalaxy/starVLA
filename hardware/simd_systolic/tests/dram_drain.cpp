#include <cstdint>
#include "dramsim3.h"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <memory>
int main(int argc,char** argv){
 assert(argc==3);std::filesystem::create_directories(argv[2]);
 unsigned readAck=0,writeAck=0;uint64_t tick=0,posted=0;
 auto m=std::unique_ptr<dramsim3::MemorySystem>(dramsim3::GetMemorySystem(argv[1],argv[2],
  [&](uint64_t){readAck++;},[&](uint64_t){writeAck++;posted=tick;}));
 assert(m->AddTransaction(0x10000,true));
 // A read forwarded from the write buffer must not count as a physical READ.
 assert(m->AddTransaction(0x10000,false));m->RequestDrain();
 while(!writeAck){m->ClockTick();tick++;assert(tick<10000);}
 assert(!m->IsDrained());
 while(!m->IsDrained()){m->ClockTick();tick++;assert(tick<10000);}
 assert(writeAck==1&&readAck==1&&m->GetWriteBursts()==1&&m->GetReadBursts()==0);
 auto drained=tick;assert(drained>posted);
 assert(m->AddTransaction(0x20000,false));m->RequestDrain();
 while(!m->IsDrained()){m->ClockTick();tick++;assert(tick<10000);}
 assert(readAck==2&&m->GetReadBursts()==1);
 std::cout<<"{\"posted_write_ack_cycle\":"<<posted<<",\"write_drain_cycle\":"<<drained
  <<",\"read_bursts\":"<<m->GetReadBursts()<<",\"write_bursts\":"<<m->GetWriteBursts()<<",\"passed\":true}\n";
 m->PrintStats();
}
