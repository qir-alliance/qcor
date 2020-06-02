#include "qcor.hpp"
// Use the pre-defined IQFT kernel
#include "qft.hpp"

using namespace qcor;

// The Oracle: T gate
__qpu__ void compositeOp(qreg q) {
  // T gate on the last qubit
  int bitIdx = q.size() - 1;
  T(q[bitIdx]);
}

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
    for (int j = 0; j < nbCalls; ++j) {
      int ctlBit = i;
      // Controlled-Oracle
      Controlled::Apply(ctlBit, compositeOp, q);
    }
  }

  // Inverse QFT on the counting qubits:
  inverseQuantumFourierTransform(q, bitPrecision);

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
