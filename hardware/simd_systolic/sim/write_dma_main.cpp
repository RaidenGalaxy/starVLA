#include "VWriteDma.h"
#include "verilated.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

static uint8_t value(unsigned i){return uint8_t((i*73+19)^(i>>5));}
int main(int argc,char**argv) {
    Verilated::commandArgs(argc,argv);VWriteDma d;std::mt19937 rng(879);
    d.clock=0;d.reset=1;d.io_command_valid=0;d.io_push_valid=0;
    d.io_request_ready=0;d.io_data_ready=0;d.io_response_valid=0;d.io_done_ready=0;
    auto edge=[&](){d.clock=0;d.eval();d.clock=1;d.eval();d.clock=0;d.eval();};
    for(unsigned i=0;i<4;++i)edge();d.reset=0;
    unsigned totalBytes=0,totalBursts=0;
    std::vector<unsigned> sizes={1,15,16,17,63,64,65,127,1024,8193};
    for(unsigned test=0;test<sizes.size()*4;++test) {
        auto count=sizes[test%sizes.size()];unsigned prefix=std::vector<unsigned>{0,1,17,63}[test/sizes.size()];
        uint64_t base=0x80000000ULL+test*0x10000+prefix;
        d.io_command_valid=1;d.io_command_bits_address=base;
        d.io_command_bits_bytes=count;d.io_command_bits_tag=test;d.eval();
        if(!d.io_command_ready)throw std::runtime_error("writer not idle");
        edge();d.io_command_valid=0;
        unsigned sent=0,stored=0,bursts=0,beats=0,delay=0,pushBytes=0;
        bool pushing=false,pendingHeader=false,ackPending=false,done=false;
        bool heldHeader=false,heldData=false;uint64_t oldAddress=0;
        unsigned oldBeat=0,oldEnable=0;uint32_t oldData[4]{};
        std::vector<uint8_t> memory(count,0);std::vector<bool> seen(count,false);
        for(unsigned cycle=0;cycle<100000 && !done;++cycle) {
            if(!pushing&&sent<count&&rng()%3!=0) {
                pushBytes=std::min(count-sent,1+unsigned(rng()%64));pushing=true;
                for(unsigned i=0;i<16;++i)d.io_push_bits_data[i]=0xffffffffu;
                for(unsigned i=0;i<pushBytes;++i) {
                    auto &word=d.io_push_bits_data[i/4];unsigned shift=8*(i%4);
                    word=(word&~(255u<<shift))|uint32_t(value(sent+i))<<shift;
                }
                d.io_push_bits_bytes=pushBytes;
            }
            d.io_push_valid=pushing;d.io_request_ready=rng()%4!=0;
            d.io_data_ready=rng()%3!=0;d.io_done_ready=rng()%3!=0;
            d.io_response_valid=ackPending&&delay==0;d.io_response_bits_id=0;
            bool error=test==sizes.size()*4-1 && bursts==1;
            d.io_response_bits_error=error;
            d.eval();
            if(heldHeader&&(!d.io_request_valid||d.io_request_bits_address!=oldAddress))
                throw std::runtime_error("stalled write header changed");
            if(heldData) {
                if(!d.io_data_valid||d.io_data_bits_beat!=oldBeat||d.io_data_bits_enable!=oldEnable)
                    throw std::runtime_error("stalled write beat changed");
                for(unsigned i=0;i<4;++i)if(oldData[i]!=d.io_data_bits_data[i])throw std::runtime_error("stalled write data changed");
            }
            heldHeader=d.io_request_valid&&!d.io_request_ready;oldAddress=d.io_request_bits_address;
            heldData=d.io_data_valid&&!d.io_data_ready;oldBeat=d.io_data_bits_beat;oldEnable=d.io_data_bits_enable;
            for(unsigned i=0;i<4;++i)oldData[i]=d.io_data_bits_data[i];
            if(d.io_request_valid&&d.io_request_ready) {
                if(pendingHeader||ackPending||d.io_request_bits_id!=0||d.io_request_bits_address!=(base&~63ULL)+64*bursts)
                    throw std::runtime_error("write burst address/ordering");
                pendingHeader=true;beats=0;++bursts;
            }
            if(d.io_data_valid&&d.io_data_ready) {
                if(!pendingHeader||d.io_data_bits_beat!=beats||d.io_data_bits_id!=0)throw std::runtime_error("write beat without header");
                for(unsigned b=0;b<16;++b) {
                    auto offset=(bursts-1)*64+beats*16+b;
                    bool valid=d.io_data_bits_enable>>b&1;
                    if(valid!=(offset>=prefix&&offset<prefix+count))throw std::runtime_error("write byte-enable failure");
                    if(valid) {
                        offset-=prefix;
                        if(seen[offset])throw std::runtime_error("duplicate byte");seen[offset]=true;
                        memory[offset]=d.io_data_bits_data[b/4]>>(8*(b%4));++stored;
                    }
                }
                if(++beats==4){pendingHeader=false;ackPending=true;delay=1+rng()%13;}
            }
            bool consume=d.io_response_valid&&d.io_response_ready;
            bool push=d.io_push_valid&&d.io_push_ready;
            if(d.io_done_valid&&d.io_done_ready) {
                if(d.io_done_bits_tag!=test||bool(d.io_done_bits_error)!=(test==sizes.size()*4-1)||ackPending||pendingHeader||sent!=count)
                    throw std::runtime_error("posted completion failure");
                done=true;
            }
            edge();
            if(push){sent+=pushBytes;pushing=false;}
            if(consume)ackPending=false;else if(delay)--delay;
        }
        if(!done||stored!=count||bursts!=(prefix+count+63)/64)throw std::runtime_error("writer timeout/count");
        for(unsigned i=0;i<count;++i)if(!seen[i]||memory[i]!=value(i))throw std::runtime_error("packed result mismatch");
        totalBytes+=count;totalBursts+=bursts;
    }
    if(d.io_payloadBytes!=totalBytes||d.io_requests!=totalBursts)throw std::runtime_error("writer counters");
    std::cout<<"{\"passed\":true,\"payload_bytes\":"<<totalBytes<<",\"write_bursts\":"<<totalBursts
             <<",\"partial_byte_enables\":true,\"unaligned_addresses\":true,\"backpressure\":true}\n";
}
