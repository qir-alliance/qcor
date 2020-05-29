#include "qcor.hpp"
// The header file which contains QFT kernel def
#include "qft.hpp"

// Entry point kernel
__qpu__ void f(qreg q) {
  const auto nQubits = q.size();
  // Add some gates
  for (int qIdx = 0; qIdx < nQubits; ++qIdx) {
    H(q[qIdx]);
  }  

  // Call qft kernel (defined in a separate header file)
  qft(q);  
  
  for (int qIdx = 0; qIdx < nQubits; ++qIdx) {
    Measure(q[qIdx]);
  }  
}

// Compile:
// qcor -o multiple_kernels -qpu qpp -shots 1024 -qrt multiple_kernels.cpp
int main(int argc, char **argv) {
  // Allocate 3 qubits
  auto q = qalloc(3);
  // Call entry-point kernel
  f(q);
  // dump the results
  q.print();
}
