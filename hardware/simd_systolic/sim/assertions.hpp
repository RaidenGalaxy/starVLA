#pragma once
#include "verilated.h"
#include <stdexcept>

// A hierarchical child's $stop can run on a parent worker. Calling exit()
// there attempts to tear down that worker's pool and can deadlock. Keep the
// stop/error flags, return through the evaluation barrier, and fail on the
// harness thread. No assertion is disabled or converted into a successful run.
inline void simdConfigureAssertions() {
    Verilated::threadContextp()->fatalOnError(false);
}

template<class Model> inline void simdEval(Model& model) {
    model.eval();
    if (Verilated::threadContextp()->gotError()) {
        throw std::runtime_error("RTL assertion stopped simulation");
    }
    if (Verilated::threadContextp()->gotFinish()) {
        throw std::runtime_error("unexpected RTL simulation finish");
    }
}
