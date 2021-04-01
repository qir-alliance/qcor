#include <qcor_qft>

// Here, we are saying this is what our QPE algorithm 
// expects as a kernel signature for user-provided 
// oracles. All oracles in this example must take a qreg and and int.
using QPEOracleSignature = KernelSignature<qreg, int>;

// Define our QPE algorithm, taking as input the 
// qreg to run on and an Oracle provided as a 
// qcor quantum kernel of the required function signature.
__qpu__ void QuantumPhaseEstimation(qreg q, QPEOracleSignature oracle) {
  // We have nQubits, the last one we use
  // as the state qubit, the others we use as the counting qubits
  const auto nQubits = q.size();
  const auto nCounting = nQubits - 1;
  const auto state_qubit_idx = nQubits - 1;

  // Put it in |1> eigenstate
  X(q[state_qubit_idx]);

  // Create uniform superposition
  for (auto i : range(nCounting)) {
    H(q[i]);
  }

  for (auto i : range(nCounting)) {
    const int nbCalls = 1 << i;
    for (auto j : range(nbCalls)) {
      oracle.ctrl(i, q, state_qubit_idx);
    }
  }

  // Run Inverse QFT, on 0:nCounting qubits
  int startIdx = 0;
  int shouldSwap = 1;
  iqft(q, startIdx, nCounting, shouldSwap);

  for (int i : range(nCounting)) {
    Measure(q[i]);
  }
}

// QPE Problem
// In this example, we demonstrate a simple QPE algorithm, i.e.
// i.e. Oracle(|State>) = exp(i*Phase)*|State>
// and we need to estimate that Phase value.
// The Oracle in this case is a T gate and the eigenstate is |1>
// i.e. T|1> = exp(i*pi/4)|1>
// We use 3 counting bits => totally 4 qubits.

// Oracle I want to consider
__qpu__ void compositeOp(qreg q, int idx) { T(q[idx]); }

int main(int argc, char **argv) {
  auto q = qalloc(4);

  // Just pass the oracle kernel to the 
  // phase estimation algorithm.
  QuantumPhaseEstimation(q, compositeOp);
  q.print();
}
