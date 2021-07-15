
// Demonstrate integration b/w QASM3 kernels with Q# Library code:
// in this case EstimateRealOverlapBetweenStates from the Standard Library (Hadamard test)
// https://github.com/microsoft/QuantumLibraries/blob/main/Standard/src/Characterization/Distinguishability.qs
// Driver to tie them all together:
// - Declare the exported QASM3 callable 
// - Use that as input to the Q# operation

// Compile with:
// qcor -qrt ftqc overlap_calc.qs kernel.qasm driver.cpp

// Note: this driver can be part of QASM3 as well, but
// we don't have the concept of Callable type in QASM3 yet,
// hence, we cannot declare the Q# kernel taking Callable argumements as extern
// in QASM3 yet.
#include "qir-types.hpp"

// QASM3 function wrapping the quantum sub-routine as a QIR Callable
extern "C" ::Callable* qasm_x__callable(); 
extern "C" ::Callable* qasm_h__callable(); 
// Q# functions:
// Compute the overlap b/w states prepared by two ansatz kernels (defined in QASM3)
extern "C" double QCOR__ComputeOverlapBetweenState__body(::Callable *, ::Callable *, int64_t /*n iters*/);

int main() {
  const double overlapped = QCOR__ComputeOverlapBetweenState__body(
      qasm_x__callable(), qasm_h__callable(), 10000);
  // Print out the results:
  // Expected: 1/sqrt(2) b/w |1> and |+> state.
  std::cout << "Overlap = " << overlapped
            << " vs. expected = " << 1.0 / std::sqrt(2.0) << "\n";
  return 0;
}