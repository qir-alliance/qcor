#include "qcor.hpp"

// Demonstrating Bell Test using multiple kernels

__qpu__ void measureAllQubits(qreg q) {
  for (int qIdx = 0; qIdx < q.size(); ++qIdx) {
    Measure(q[qIdx]);
  }  
}

// Entangle all qubit in a qubit register with a master qubit
__qpu__ void entangleAll(qreg q, int masterBitIdx) {
  for (int qIdx = 0; qIdx < masterBitIdx; ++qIdx) {
    CNOT(q[masterBitIdx], q[qIdx]);
  }  
  
  for (int qIdx = masterBitIdx + 1; qIdx < q.size(); ++qIdx) {
    CNOT(q[masterBitIdx], q[qIdx]);
  }  
}

// Entry point kernel
__qpu__ void bellTest(qreg qBits) {
  // Add some gates
  // Entangle the *middle* qubit with all other qubits
  int masterBit = qBits.size() / 2;
  // Hadamard
  H(qBits[masterBit]);
  // Entangle
  entangleAll(qBits, masterBit);
  
  // Measure all qubits
  measureAllQubits(qBits);
}

// Compile:
// qcor -o multiple_kernels -qpu qpp -shots 1024 -qrt multiple_kernels.cpp
int main(int argc, char **argv) {
  // Allocate 7 qubits: 
  // i.e. Hadamard on q[3] and entangle q[3] with other qubits.
  auto q = qalloc(7);
  // Call entry-point kernel
  bellTest(q);
  // dump the results
  // Expect: ~50-50 for "0000000" and "1111111"
  q.print();
}
