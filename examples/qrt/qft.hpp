// Example code structure whereby quantum kernels are defined
// in separate header files which can be included into a cpp file
// which is compiled by QCOR.
#pragma once

#include "qcor.hpp"

// QFT kernel
__qpu__ void qft(qreg q) {
  // Local Declarations
  const auto nQubits = q.size();

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