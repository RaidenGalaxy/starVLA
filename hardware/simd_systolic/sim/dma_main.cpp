#include "VReadDma.h"
#include "verilated.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

struct Return {unsigned id,beat;uint64_t address;bool error;};
static std::array<uint32_t,4> payload(uint64_t address,unsigned beat) {
    auto n=address/16+beat;
    return {uint32_t(n),uint32_t(n>>32),uint32_t(n^0xa5527093),uint32_t(~n)};
}
int main(int argc,char**argv) {
    Verilated::commandArgs(argc,argv);VReadDma d;
    std::mt19937 rng(831);unsigned maximum=0,totalRequests=0,totalLoads=0;
    d.clock=0;d.reset=1;d.io_command_valid=0;d.io_response_valid=0;
    d.io_request_ready=0;d.io_load_ready=0;d.io_done_ready=0;
    auto edge=[&](){d.clock=0;d.eval();d.clock=1;d.eval();d.clock=0;d.eval();};
    for(unsigned i=0;i<4;++i)edge();d.reset=0;
    std::vector<unsigned> sizes={1,3,4,5,127,512,1024,2048,33};
    for(unsigned test=0;test<sizes.size();++test) {
        unsigned count=sizes[test],bank=count>1024?1:test%3;
        unsigned destination=count>=1024?0:7;
        uint64_t base=0x10000000ULL+test*0x10000;
        d.io_command_bits_source=base;d.io_command_bits_bank=bank;
        d.io_command_bits_destination=destination;d.io_command_bits_beats=count;
        d.io_command_bits_tag=test;d.io_command_valid=1;
        d.eval();if(!d.io_command_ready)throw std::runtime_error("DMA not idle");
        edge();d.io_command_valid=0;
        std::vector<Return> waiting;std::map<unsigned,bool> seen;
        bool responseActive=false,done=false,held=false;
        Return current{};uint64_t heldAddress=0;unsigned heldId=0,requests=0,loads=0;
        bool injectError=test==sizes.size()-1;
        for(unsigned cycle=0;cycle<100000 && !done;++cycle) {
            d.io_request_ready=rng()%4!=0;d.io_load_ready=rng()%3!=0;
            d.io_done_ready=cycle>100 && rng()%3!=0;
            if(!responseActive && cycle>80 && !waiting.empty() && rng()%4!=0) {
                unsigned i=rng()%waiting.size();current=waiting[i];
                waiting.erase(waiting.begin()+i);responseActive=true;
            }
            d.io_response_valid=responseActive;
            if(responseActive) {
                d.io_response_bits_id=current.id;d.io_response_bits_beat=current.beat;
                d.io_response_bits_error=current.error;
                auto data=payload(current.address,current.beat);
                for(unsigned i=0;i<4;++i)d.io_response_bits_data[i]=data[i];
            }
            d.eval();maximum=std::max(maximum,unsigned(d.io_outstanding));
            if(d.io_outstanding>32)throw std::runtime_error("DMA credit overflow");
            if(held && (!d.io_request_valid || d.io_request_bits_address!=heldAddress || d.io_request_bits_id!=heldId))
                throw std::runtime_error("DMA changed a stalled request");
            held=d.io_request_valid&&!d.io_request_ready;
            heldAddress=d.io_request_bits_address;heldId=d.io_request_bits_id;
            if(d.io_request_valid&&d.io_request_ready) {
                auto addr=d.io_request_bits_address;
                if(addr!=base+requests*64)throw std::runtime_error("DMA request address");
                for(unsigned b=0;b<4;++b)
                    waiting.push_back({unsigned(d.io_request_bits_id),b,addr,injectError&&requests==0&&b==2});
                ++requests;
            }
            if(d.io_load_valid&&d.io_load_ready) {
                unsigned offset=(current.address-base)/16+current.beat;
                if(!responseActive || current.error || offset>=count ||
                   d.io_load_bits_address!=destination+offset || d.io_load_bits_bank!=bank || d.io_load_bits_strobe!=65535)
                    throw std::runtime_error("DMA local word address or padding");
                if(seen[offset])throw std::runtime_error("DMA duplicate write");
                seen[offset]=true;
                auto data=payload(current.address,current.beat);
                for(unsigned i=0;i<4;++i)if(data[i]!=d.io_load_bits_data[i])throw std::runtime_error("DMA payload mismatch");
                ++loads;
            }
            bool consumed=d.io_response_valid&&d.io_response_ready;
            if(d.io_done_valid&&d.io_done_ready) {
                if(d.io_done_bits_tag!=test || bool(d.io_done_bits_error)!=injectError ||
                   !waiting.empty() || responseActive || d.io_outstanding)
                    throw std::runtime_error("DMA completed before data drain");
                done=true;
            }
            edge();if(consumed)responseActive=false;
        }
        if(!done || requests!=(count+3)/4 || loads!=count-unsigned(injectError))
            throw std::runtime_error("DMA count/timeout failure");
        totalLoads+=loads;totalRequests+=requests;
    }
    if(maximum!=32 || d.io_requests!=totalRequests || d.io_deliveredBytes!=16*totalLoads)
        throw std::runtime_error("DMA performance counter failure");
    std::cout<<"{\"passed\":true,\"maximum_credits\":"<<maximum<<",\"read_bursts\":"<<totalRequests
             <<",\"load_beats\":"<<totalLoads<<",\"reordered_responses\":true,\"backpressure\":true}\n";
}
