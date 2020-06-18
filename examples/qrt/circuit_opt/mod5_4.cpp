// Demonstrate loading openqasm file with
// #include call, the
// qcor::measure_all API call, and programmatically
// optimizing a quantum kernel.
//
// run this with
// qcor -qrt -qpu aer mod5_4.cpp
// ./a.out

#include "qcor.hpp"

__qpu__ void mod5_4(qreg q) {
  using qcor::openqasm;

#include "mod5_4.qasm"
}

int main() {

  // Allocate the qubits
  auto q = qalloc(5);

  // This kernel is unmeasured,
  // add measures to them
  auto measured = qcor::measure_all(mod5_4, q);

  std::cout << "Number of Gates Before: "
            << qcor::n_instructions(measured, q) << "\n";

  // Apply some circuit optimizations
  auto optimized_kernel = qcor::apply_transformations(
      measured, {"circuit-optimizer", "rotation-folding"}, q);

  std::cout << "Number of Gates After: "
            << qcor::n_instructions(optimized_kernel, q) << "\n";
    
  optimized_kernel(q);
  q.print();
}