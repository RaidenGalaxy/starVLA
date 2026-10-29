#pragma once
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>
template<class W> void setWide(W& x,const std::string& text,int words=16) {
 for(int i=0;i<words;i++) {int end=int(text.size())-i*8;
  x[i]=end>0?std::stoul(text.substr(std::max(0,end-8),std::min(end,8)),nullptr,16):0;}
}
template<class W> std::string wideHex(const W& x,int words=16) {
 std::ostringstream s;s<<std::hex<<std::setfill('0');
 for(int i=words-1;i>=0;i--)s<<std::setw(8)<<x[i];return s.str();
}
