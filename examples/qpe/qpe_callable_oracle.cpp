#include <qcor_qft>

using QPEOracle = CallableKernel<qreg, int>;

__qpu__ void QuantumPhaseEstimation(qreg q, QPEOracle oracle) {
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
      int ctlBit = i;
      // Will be fixing this parent_kernel thing...
      oracle.ctrl(ctlBit, q, state_qubit_idx);
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
  // very cool, implicit conversion works here
  // so you can just pass the kernel function
  QuantumPhaseEstimation(q, compositeOp);
  q.print();
}
