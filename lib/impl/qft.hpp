// Example code structure whereby quantum kernels are defined
// in separate header files which can be included into a cpp file
// which is compiled by QCOR.
#pragma once

#include "qalloc.hpp"

// Generic QFT and IQFT algorithms
// Input:
// (1) Qubit register (type qreg)
// (2) The start qubit index and the number of qubits (type integer)
// Note: these two parameters allow us to operate the QFT/IQFT on 
// a contiguous subset of qubits of the register.
// If we want to use the entire register, 
// just pass 0 and number of qubits in the register, respectively.
// (3) shouldSwap flag (as integer):
// If 0, no Swap gates will be added.
// Otherwise, Swap gates are added at the end (QFT) and beginning (IQFT).
__qpu__ void qft(qreg q, int startIdx, int nbQubits, int shouldSwap) {
  for (int qIdx = nbQubits - 1; qIdx >= 0; --qIdx) {
    auto shiftedBitIdx = qIdx + startIdx;
    H(q[shiftedBitIdx]);
    for (int j = qIdx - 1; j >= 0; --j) {
      const double theta = M_PI/std::pow(2.0, qIdx - j);
      auto targetIdx = j + startIdx;
      CPhase(q[shiftedBitIdx], q[targetIdx], theta);
    }
  }

  // A *hacky* way to do conditional (convert to a for loop)
  int swapCount = (shouldSwap == 0) ? 0 : 1;
  for (int count = 0; count < swapCount; ++count) {
    for (int qIdx = 0; qIdx < nbQubits/2; ++qIdx) {
      Swap(q[startIdx + qIdx], q[startIdx + nbQubits - qIdx - 1]);
    }
  }
}

__qpu__ void iqft(qreg q, int startIdx, int nbQubits, int shouldSwap) {
  int swapCount = (shouldSwap == 0) ? 0 : 1;
  for (int count = 0; count < swapCount; ++count) {
    // Swap qubits
    for (int qIdx = 0; qIdx < nbQubits/2; ++qIdx) {
      Swap(q[startIdx + qIdx], q[startIdx + nbQubits - qIdx - 1]);
    }
  }

  for (int qIdx = 0; qIdx < nbQubits - 1; ++qIdx) {
    H(q[startIdx + qIdx]);
    int j = qIdx + 1;
    for (int y = qIdx; y >= 0; --y) {
      const double theta = -M_PI/std::pow(2.0, j - y);
      CPhase(q[startIdx + j], q[startIdx + y], theta);
    }
  }

  H(q[startIdx + nbQubits - 1]);
}