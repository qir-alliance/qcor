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

  // Allocate the qubits, just use an upper bound number of qubits.
  auto q = qalloc(30);

  // Run the kernel
  testKernel(q);
}
#endif