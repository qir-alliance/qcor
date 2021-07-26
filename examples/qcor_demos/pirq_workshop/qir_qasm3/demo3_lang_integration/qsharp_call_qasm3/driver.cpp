
// Driver to tie them all together:
// - Declare the exported QASM3 callable 
// - Use that as input to the Q# operation

// Compile with:
// qcor -qrt ftqc op_takes_callable.qs kernel.qasm driver.cpp -v

// Note: this driver can be part of QASM3 as well, but
// we don't have the concept of Callable type in QASM3 yet,
// hence, we cannot declare the Q# kernel taking Callable argumements as extern
// in QASM3 yet.
#include "qir-types.hpp"

// QASM3 function wrapping the quantum sub-routine as a QIR Callable
extern "C" ::Callable* qasm_x__callable(); 

// Q# functions:
// Apply Controlled version of a Callable (X gate in QASM3)
// This will just be a Bell experiment.
extern "C" void QCOR__ApplyControlledKernel__body(::Callable *);

int main() {
  // Get the callable (QASM3)
  auto qasm3_callable = qasm_x__callable();
  // Pass it to Q#
  // std::cout << "Apply the functor to each qubit:\n";
  // QCOR__ApplyKernelToEachQubit__body(qasm3_callable);

  // Run Bell experiment:
  constexpr int COUNT = 100;
  std::cout << "Apply controlled functor(Bell test):\n";
  for (int i = 0; i < COUNT; ++i) {
    std::cout << "Run " << i + 1 << ":\n";
    QCOR__ApplyControlledKernel__body(qasm3_callable);
  }
  return 0;
}