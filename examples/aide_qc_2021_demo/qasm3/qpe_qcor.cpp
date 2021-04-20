#include <qcor_qft>

// QPE Problem
// In this example, we demonstrate a simple QPE algorithm, i.e.
// i.e. Oracle(|State>) = exp(i*Phase)*|State>
// and we need to estimate that Phase value.
// The Oracle in this case is a T gate and the eigenstate is |1>
// i.e. T|1> = exp(i*pi/4)|1>
// We use 3 counting bits => totally 4 qubits.

// Our qpe kernel requires oracles with 
// the following signature.
using QPEOracleSignature = KernelSignature<qubit>;

__qpu__ void qpe(qreg q, QPEOracleSignature oracle) {
  // Extract the counting qubits and the state qubit
  auto counting_qubits = q.extract_range({0,3});
  // could also do this...
  // auto counting_qubits = q.extract_qubits({0,1,2});
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

  // Run Inverse QFT on counting qubits
  iqft(counting_qubits);

  // Measure the counting qubits
  Measure(counting_qubits);
}

// Oracle I want to consider
__qpu__ void oracle(qubit q) { T(q); }

int main(int argc, char **argv) {
  auto q = qalloc(4);
  qpe(q, oracle);
  q.print();
}
