// Example code structure whereby quantum kernels are defined
// in separate header files which can be included into a cpp file
// which is compiled by QCOR.
#pragma once

#include "qalloc.hpp"

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
    for (int qIdx = 0; qIdx < (nbQubits - 1)/2; ++qIdx) {
      Swap(q[startIdx + qIdx], q[startIdx + nbQubits - qIdx - 1]);
    }
  }
}

__qpu__ void iqft(qreg q, int startIdx, int nbQubits, int shouldSwap) {
  int swapCount = (shouldSwap == 0) ? 0 : 1;
  for (int count = 0; count < swapCount; ++count) {
    // Swap qubits
    for (int qIdx = 0; qIdx < (nbQubits - 1)/2; ++qIdx) {
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

// QFT kernel:
// Input: Qubit register and the max qubit index for the QFT,
// i.e. allow us to do QFT on a subset of the register [0, maxBitIdx)
// __qpu__ void qft(qreg q, int maxBitIdx) {
//   // Local Declarations
//   const auto nQubits = maxBitIdx;

//   for (int qIdx = 0; qIdx < nQubits; ++qIdx) {
//     auto bitIdx = nQubits - qIdx - 1;
//     H(q[bitIdx]);
//     for (int j = 0; j < bitIdx; ++j) {
//       const double theta = M_PI/std::pow(2.0, bitIdx - j);
//       CPhase(q[j], q[bitIdx], theta);
//     }
//   }

//   // Swap qubits
//   for (int qIdx = 0; qIdx < nQubits / 2; ++qIdx) {
//     Swap(q[qIdx], q[nQubits - qIdx - 1]);
//   }
// }

// // Inverse QFT
// __qpu__ void iqft(qreg q, int maxBitIdx) {
//   // Local Declarations
//   const auto nQubits = maxBitIdx;
//   // Swap qubits
//   for (int qIdx = 0; qIdx < nQubits / 2; ++qIdx) {
//     Swap(q[qIdx], q[nQubits - qIdx - 1]);
//   }

//   for (int qIdx = 0; qIdx < nQubits - 1; ++qIdx) {
//     H(q[qIdx]);
//     for (int j = 0; j < qIdx + 1; ++j) {
//       const double theta = -M_PI/std::pow(2.0, qIdx + 1 - j);
//       CPhase(q[j], q[qIdx + 1], theta);
//     }
//   }

//   H(q[nQubits - 1]);
// }