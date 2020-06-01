// Example code structure whereby quantum kernels are defined
// in separate header files which can be included into a cpp file
// which is compiled by QCOR.
#pragma once

#include "qcor.hpp"

// QFT kernel:
// Input: Qubit register and the max qubit index for the QFT,
// i.e. allow us to do QFT on a subset of the register [0, maxBitIdx)
__qpu__ void quantumFourierTransform(qreg q, int maxBitIdx) {
  // Local Declarations
  const auto nQubits = maxBitIdx;

  for (int qIdx = 0; qIdx < nQubits; ++qIdx) {
    auto bitIdx = nQubits - qIdx - 1;
    H(q[bitIdx]);
    for (int j = 0; j < bitIdx; ++j) {
      const double theta = M_PI/std::pow(2.0, bitIdx - j);
      CPhase(q[j], q[bitIdx], theta);
    }
  }

  // Swap qubits
  for (int qIdx = 0; qIdx < nQubits / 2; ++qIdx) {
    Swap(q[qIdx], q[nQubits - qIdx - 1]);
  }
}

// Inverse QFT
__qpu__ void inverseQuantumFourierTransform(qreg q, int maxBitIdx) {
  // Local Declarations
  const auto nQubits = maxBitIdx;
  // Swap qubits
  for (int qIdx = 0; qIdx < nQubits / 2; ++qIdx) {
    Swap(q[qIdx], q[nQubits - qIdx - 1]);
  }

  for (int qIdx = 0; qIdx < nQubits - 1; ++qIdx) {
    H(q[qIdx]);
    for (int j = 0; j < qIdx + 1; ++j) {
      const double theta = -M_PI/std::pow(2.0, qIdx + 1 - j);
      CPhase(q[j], q[qIdx + 1], theta);
    }
  }

  H(q[nQubits - 1]);
}