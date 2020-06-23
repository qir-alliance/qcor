// Demonstrate loading openqasm file with
// #include call, the
// qcor::measure_all API call, and programmatically
// optimizing a quantum kernel.
//
// run this with
// qcor -qrt -qpu aer grover.cpp
// ./a.out

#include "qcor.hpp"

__qpu__ void grover_5(qreg q) {
  using qcor::openqasm;

#include "grover_5.qasm"
}

int main() {

  // Allocate the qubits
  auto q = qalloc(9);

  // This kernel is unmeasured,
  // add measures to them
  auto measure_grov = qcor::measure_all(grover_5, q);

  std::cout << "Number of Gates Before: "
            << qcor::n_instructions(measure_grov, q) << "\n";

  // Apply some circuit optimizations
  auto optimized_kernel = qcor::apply_transformations(
      measure_grov, {"circuit-optimizer", "rotation-folding"}, q);

  std::cout << "Number of Gates After: "
            << qcor::n_instructions(optimized_kernel, q) << "\n";
}