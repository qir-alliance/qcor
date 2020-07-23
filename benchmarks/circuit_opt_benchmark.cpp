// Run a suite of circuit optimization benchmarks.
#include "qcor.hpp"

// Need to pass -DTEST_SOURCE_FILE=\"test_case_filename\" and -I resources/ to
// the qcor compiler: e.g.
// qcor -qpu qpp -DTEST_SOURCE_FILE=\"adder_8.qasm\" -I resources/
// circuit_opt_benchmark.cpp

#ifdef TEST_SOURCE_FILE
__qpu__ void testKernel(qreg q) {
  using qcor::openqasm;
#include TEST_SOURCE_FILE
}

int main() {

  // Allocate just 1 qubit, we don't actually want to run the simulation.
  auto q = qalloc(1);

  // Run the kernel
  testKernel(q);
}
#endif