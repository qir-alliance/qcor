#include "qcor.hpp"

// Example of Quantum Phase Estimation circuit using QCOR runtime.
// Compile with:
// qcor -o qpe -qpu qpp -shots 1024 -qrt qpe_example_qrt.cpp
// ----------------------------------------------------------------
// In this example, we demonstrate a simple QPE algorithm, i.e.
// i.e. Oracle(|State>) = exp(i*Phase)*|State>
// and we need to estimate that Phase value.
// The Oracle in this case is a T gate and the eigenstate is |1>
// i.e. T|1> = exp(i*pi/4)|1>
// We use 3 counting bits => totally 4 qubits.
__qpu__ void QuantumPhaseEstimation(qreg q) {
  const auto nQubits = q.size();
  // Last qubit is the eigenstate of the unitary operator 
  // hence, prepare it in |1> state
  X(q[nQubits - 1]);

  // Apply Hadamard gates to the counting qubits:
  for (int qIdx = 0; qIdx < nQubits - 1; ++qIdx) {
    H(q[qIdx]);
  }

  // Apply Controlled-Oracle: in this example, Oracle is T gate;
  // i.e. Ctrl(T) = CPhase(pi/4)
  const auto bitPrecision = nQubits - 1;
  for (int32_t i = 0; i < bitPrecision; ++i) {
    const int nbCalls = 1 << i;
    // Ctrl(T) = CPhase(pi/4)
    for (int j = 0; j < nbCalls; ++j) {
      CPhase(q[i], q[nQubits - 1], M_PI_4);
    }
  }

  // Inverse QFT on the counting qubits:
  const int startIdx = bitPrecision / 2 - 1;
  for (int qIdx = startIdx; qIdx >= 0; --qIdx) {
    Swap(q[qIdx], q[bitPrecision - qIdx - 1]);
  }

  for (int qIdx = 0; qIdx < bitPrecision; ++qIdx) {
    for (int j = 0; j < qIdx; ++j) {
      const double theta = -M_PI/std::pow(2.0, qIdx - j);
      CPhase(q[j], q[qIdx], theta);
    }
    H(q[qIdx]);
  }

  // Measure counting qubits
  for (int qIdx = 0; qIdx < bitPrecision; ++qIdx) {
    Measure(q[qIdx]);
  }
}

int main(int argc, char **argv) {
  // Allocate 4 qubits, i.e. 3-bit precision
  auto q = qalloc(4);
  QuantumPhaseEstimation(q);
  // dump the results
  // EXPECTED: only "100" bitstring
  q.print();
}
