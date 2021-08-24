#include "qir_nisq_kernel_utils.hpp"
#include "qcor.hpp"

// Compile:
// qcor iqft.o qpe.cpp -shots 1024 
using QPEOracleSignature = KernelSignature<qubit>;

qcor_import_qasm3_kernel(py_qiskit_iqft);

__qpu__ void qpe(qreg q, QPEOracleSignature oracle) {
  // Extract the counting qubits and the state qubit
  auto counting_qubits = q.extract_range({0,3});
  auto state_qubit = q[3];
  // Put it in |1> eigenstate
  X(state_qubit);
  
  // Create uniform superposition on all 3 qubits
  H(counting_qubits);

  // run ctr-oracle operations
  for (auto i : range(counting_qubits.size())) {
    const int nbCalls = 1 << i;
    for (auto j : range(nbCalls)) {
      oracle.ctrl(counting_qubits[i], state_qubit);
    }
  }

  // Run Inverse QFT on counting qubits from qiskit
  py_qiskit_iqft(counting_qubits);

  // Measure the counting qubits
  Measure(counting_qubits);
}

// Oracle to consider
__qpu__ void oracle(qubit q) { T(q); }

int main(int argc, char **argv) {
  auto q = qalloc(4);
  // Run
  qpe(q, oracle);
  q.print();
}
