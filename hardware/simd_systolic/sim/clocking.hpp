#pragma once
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>

inline uint64_t simdClockPhase(const char* name,uint64_t period) {
    const auto* value=std::getenv(name);
    if(!value)return 0;
    std::string text(value);std::size_t used=0;
    const auto phase=std::stoull(text,&used);
    if(used!=text.size()||phase>=period)throw std::runtime_error("invalid clock phase: "+std::string(name));
    return phase;
}
