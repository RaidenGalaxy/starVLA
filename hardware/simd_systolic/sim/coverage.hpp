#pragma once
#include <cstdlib>
#include <filesystem>
#include "verilated.h"
#if VM_COVERAGE
#include "verilated_cov.h"
#endif

// The destination is selected separately for every test invocation. Counts
// from failed runs are raw diagnostics and are never accepted as passing.
inline void simdWriteCoverage() {
#if VM_COVERAGE
  static bool written=false;
  const char* path=std::getenv("SIMD_CODE_COVERAGE");
  if(!written&&path&&*path){
   auto parent=std::filesystem::path(path).parent_path();
   if(!parent.empty())std::filesystem::create_directories(parent);
   Verilated::threadContextp()->coveragep()->write(path);
   written=true;
  }
#endif
}
struct SimdCoverageSave {
 ~SimdCoverageSave(){simdWriteCoverage();}
};
