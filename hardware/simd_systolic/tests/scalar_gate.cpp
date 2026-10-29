#include "VGateDut.h"
#include "verilated.h"
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

struct Vector {uint64_t a,b,c,result;unsigned op,flags;};
int main(int argc,char**argv) {
    if(argc<3||argc>4)throw std::runtime_error("gate-test vectors stages [free|elastic]");
    bool freeRunning=argc==4&&std::string(argv[3])=="free";
    unsigned stages=std::stoul(argv[2]);
    std::ifstream input(argv[1]);std::deque<Vector> vectors,pending;
    std::string a,b,c,result;unsigned op,flags;
    while(input>>a>>b>>c>>op>>result>>flags)
        vectors.push_back({std::stoull(a,nullptr,16),std::stoull(b,nullptr,16),std::stoull(c,nullptr,16),std::stoull(result,nullptr,16),op,flags});
    VGateDut dut;dut.clock=0;dut.ce=1;dut.a=0;dut.b=0;dut.c=0;dut.op=0;
    auto edge=[&](){dut.clock=0;dut.eval();dut.clock=1;dut.eval();};
    for(unsigned i=0;i<stages;++i)edge();
    unsigned checked=0,expected=vectors.size();std::mt19937 rng(129);
    while(checked<expected) {
        dut.ce=freeRunning||(rng()%4)!=0;
        Vector v{};
        if(!vectors.empty())v=vectors.front();
        dut.a=v.a;dut.b=v.b;dut.c=v.c;dut.op=v.op;
        edge();
        if(dut.ce) {
            pending.push_back(v);
            if(!vectors.empty())vectors.pop_front();
            if(pending.size()>=stages) {
                auto e=pending.front();pending.pop_front();
                if(dut.result!=e.result || dut.flags!=e.flags)
                    throw std::runtime_error("mapped gate-level numeric/latency mismatch at "+std::to_string(checked));
                ++checked;
            }
        }
    }
    std::cout<<"{\"gate_vectors\":"<<checked<<",\"passed\":true}\n";
}
