#include "VAsyncFifo.h"
#include "verilated.h"
#include <algorithm>
#include <cstdint>
#include <deque>
#include <iostream>
#include <random>
#include <stdexcept>

int main(int argc,char**argv) {
 try {
    const uint64_t wp=argc>1?std::stoull(argv[1]):6;
    const uint64_t rp=argc>2?std::stoull(argv[2]):10;
    const uint64_t phase=argc>3?std::stoull(argv[3]):0;
    if(wp<2||rp<2||wp%2||rp%2||phase>=rp)throw std::runtime_error("invalid FIFO clock periods/phase");
    VAsyncFifo d;std::mt19937 rng(731);std::deque<uint32_t> expected;
    d.wclk=0;d.rclk=0;d.wrst=0;d.rrst=0;d.wvalid=0;d.rready=0;d.eval();
    d.wrst=1;d.rrst=1;d.eval();
    uint32_t sent=0,received=0,dropped=0;uint64_t wt=0,rt=phase;
    const auto span=std::max(wp,rp),released=64*span;
    uint64_t resetEnd=0;bool didReset=false,releasedInitial=false;
    uint64_t fullClocks=0,emptyClocks=0,writeStalls=0,readStalls=0,steps=0;
    while(received+dropped<10000){
        auto t=std::min(wt,rt);
        if(!releasedInitial)t=std::min(t,released);
        if(resetEnd)t=std::min(t,resetEnd);
        if(++steps>4000000)throw std::runtime_error("CDC timeout");
        if(!releasedInitial&&t==released){d.wrst=0;d.rrst=0;releasedInitial=true;}
        if(resetEnd&&t==resetEnd){d.wrst=0;d.rrst=0;resetEnd=0;}
        if(!didReset&&received>=1024&&!expected.empty()){
            d.wrst=1;d.rrst=1;d.wvalid=0;d.rready=0;
            dropped+=expected.size();expected.clear();didReset=true;resetEnd=t+5*span+1;
        }
        const bool wr=t==wt&&!d.wclk,rd=t==rt&&!d.rclk;
        if(t==wt&&d.wclk&&!d.wrst){
            if(!d.wvalid){d.wvalid=sent<10000&&rng()%4!=0;d.wdata=sent;}
        }
        if(t==rt&&d.rclk&&!d.rrst)d.rready=t>released+32*wp&&rng()%3!=0;
        d.eval();
        const bool push=wr&&!d.wrst&&d.wvalid&&d.wready;
        const bool pop=rd&&!d.rrst&&d.rvalid&&d.rready;
        if(wr&&!d.wrst){fullClocks+=!d.wready;writeStalls+=d.wvalid&&!d.wready;}
        if(rd&&!d.rrst){emptyClocks+=!d.rvalid;readStalls+=d.rvalid&&!d.rready;}
        if(pop){
            if(expected.empty()||expected.front()!=d.rdata)throw std::runtime_error("CDC data/order failure");
            expected.pop_front();++received;
        }
        if(push){expected.push_back(sent);++sent;}
        if(t==wt){d.wclk=!d.wclk;wt+=wp/2;}
        if(t==rt){d.rclk=!d.rclk;rt+=rp/2;}
        d.eval();
        if(push)d.wvalid=0;
    }
    if(sent!=10000||received+dropped!=sent||!expected.empty()||!didReset||!dropped||!fullClocks||!emptyClocks||!writeStalls||!readStalls)
        throw std::runtime_error("missing FIFO reset/full/empty/backpressure coverage");
    std::cout<<"{\"cdc_words\":"<<received<<",\"reset_discarded_words\":"<<dropped
     <<",\"write_period\":"<<wp<<",\"read_period\":"<<rp<<",\"read_phase\":"<<phase
     <<",\"full_clocks\":"<<fullClocks<<",\"empty_clocks\":"<<emptyClocks
     <<",\"write_stalls\":"<<writeStalls<<",\"read_stalls\":"<<readStalls
     <<",\"pointer_wraps\":"<<sent/16<<",\"midflight_reset\":true,\"passed\":true}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
