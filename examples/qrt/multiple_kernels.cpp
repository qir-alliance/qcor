// The header file which contains QFT kernel def
#include "qft.hpp"

// Entry point kernel
__qpu__ void f(qreg q) {
  const auto nQubits = q.size();
  // Add some gates
  X(q[1]);

  // Call qft kernel (defined in a separate header file)
  qft(q, nQubits);  
  
  // Inverse QFT:
  iqft(q, nQubits);
  
  // Measure all qubits
  for (int qIdx = 0; qIdx < nQubits; ++qIdx) {
    Measure(q[qIdx]);
  }  
}

// Compile:
// qcor -o multiple_kernels -qpu qpp -shots 1024 -qrt multiple_kernels.cpp
// Execute:
// ./multiple_kernels
// Expected: "010" state (all shots) since we do X(q[1]) the QFT->IQFT (identity)
int main(int argc, char **argv) {
  // Allocate 3 qubits
  auto q = qalloc(3);
  // Call entry-point kernel
  f(q);
  // dump the results
  q.print();
}
