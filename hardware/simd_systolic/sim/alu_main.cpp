#include "VSimdAlu.h"
#include "verilated.h"
#include "coverage.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct Vector { unsigned fmt,op,mask,ctx,tag; uint64_t a,b; std::string acc; };
template<class W> void setWide(W& x,const std::string& text,int words=16) {
 for(int i=0;i<words;i++) {
   const int end=int(text.size())-i*8;
   x[i]=end>0 ? std::stoul(text.substr(std::max(0,end-8),std::min(end,8)),nullptr,16) : 0;
 }
}
template<class W> std::string wideHex(const W& x,int words=16) {
 std::ostringstream s; s<<std::hex<<std::setfill('0');
 for(int i=words-1;i>=0;i--) s<<std::setw(8)<<x[i];
 return s.str();
}
int main(int argc,char** argv) {
 Verilated::commandArgs(argc,argv);
 if(argc<2) {std::cerr<<"usage: alu_sim vectors.txt [stall_seed]\n";return 2;}
 std::ifstream f(argv[1]); if(!f) return 2;
 std::vector<Vector> input; Vector v;std::string a,b,mask;
 while(f>>v.fmt>>v.op>>a>>b>>v.acc>>mask>>v.ctx>>v.tag) {
   v.a=std::stoull(a,nullptr,16);v.b=std::stoull(b,nullptr,16);
   v.mask=std::stoul(mask,nullptr,16);input.push_back(v);
 }
 VSimdAlu d; SimdCoverageSave coverage;d.clock=0;d.reset=1;d.io_in_valid=0;d.io_out_ready=1;
 for(int i=0;i<4;i++){d.clock=0;d.eval();d.clock=1;d.eval();} d.reset=0;
 std::mt19937 rng(argc>2?std::stoul(argv[2]):0);
 const bool stalls=argc>2;std::size_t sent=0,received=0;
 const bool idleGaps=std::getenv("SIMD_ALU_IDLE_GAPS")!=nullptr;
 unsigned idle=idleGaps?8:0;
 std::vector<uint64_t> issue(input.size());
 for(uint64_t cycle=0;received<input.size();cycle++) {
   if(cycle>1000+input.size()*100) {std::cerr<<"timeout\n";return 3;}
   d.clock=0;d.io_out_ready=!stalls || rng()%4!=0;
   // Once valid is asserted the transaction remains stable until accepted.
   d.io_in_valid=sent<input.size() && idle==0;
   if(sent<input.size()) {
     auto& q=input[sent];d.io_in_bits_fmt=q.fmt;d.io_in_bits_op=q.op;
     d.io_in_bits_a=q.a;d.io_in_bits_b=q.b;setWide(d.io_in_bits_acc,q.acc);
     d.io_in_bits_mask=q.mask;d.io_in_bits_ctx=q.ctx;d.io_in_bits_tag=q.tag;
   }
   d.eval();
   const bool accept=d.io_in_valid&&d.io_in_ready;
   if(d.io_out_valid&&d.io_out_ready) {
     auto tag=d.io_out_bits_tag;
     std::cout<<std::dec<<tag<<' '<<wideHex(d.io_out_bits_data)<<' '
       <<std::hex<<unsigned(d.io_out_bits_flags)<<' '<<unsigned(d.io_out_bits_overflow)<<' '
       <<unsigned(d.io_out_bits_illegal)<<std::dec<<' '<<issue.at(tag)<<' '<<cycle<<'\n';
     received++;
   }
   if(idle)--idle;
   if(accept) {issue.at(input[sent].tag)=cycle;sent++;if(idleGaps && sent%31==0)idle=5;}
   d.clock=1;d.eval();
 }
 simdWriteCoverage();d.final();return 0;
}
