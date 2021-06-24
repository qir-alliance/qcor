#pragma once
#include <qalloc>
#include <vector>
#include "qcor_observable.hpp"

#ifdef _QCOR_FTQC_RUNTIME 
namespace ftqc {
__qpu__ void reset_all(qreg q) {
  for (int i = 0; i < q.size(); ++i) {
    if (Measure(q[i])) {
      X(q[i]);
    }
  }
}

// FTQC "sync" Pauli measurement: returns the parity output
__qpu__ void measure_basis(qreg q, std::vector<qcor::Operator> bases,
                      int &out_parity) {
  int oneCount = 0;
  for (int i = 0; i < bases.size(); ++i) {
    auto pauliOp = bases[i];
    const std::string pauliStr = pauliOp.toString().substr(6);
    const auto bitIdx = std::stoi(pauliStr.substr(1));
    // TODO: fix XASM compiler to handle char literal
    if (pauliStr.rfind("X", 0) == 0) {
      H(q[bitIdx]);
    }

    if (pauliStr.rfind("Y", 0) == 0) {
      Rx(q[bitIdx], M_PI_2);
    }
    if (Measure(q[bitIdx])) {
      oneCount++;
    }
  }
  out_parity = oneCount - 2 * (oneCount / 2);
}

// Measure the given Pauli operator using an explicit scratch qubit to perform the measurement.
__qpu__ void measure_basis_with_scratch(qreg q, int scratchQubit,
                                        std::vector<qcor::Operator> bases,
                                        int &out_result) {
  H(q[scratchQubit]);
  for (int i = 0; i < bases.size(); ++i) {
    auto pauliOp = bases[i];
    const std::string pauliStr = pauliOp.toString().substr(6);
    const auto bitIdx = std::stoi(pauliStr.substr(1));
    // Pauli-X
    if (pauliStr.rfind("X", 0) == 0) {
      CX(q[scratchQubit], q[bitIdx]);
    }
    // Pauli-Y
    if (pauliStr.rfind("Y", 0) == 0) {
      CY(q[scratchQubit], q[bitIdx]);
    }
    // Pauli-Z
    if (pauliStr.rfind("Z", 0) == 0) {
      CZ(q[scratchQubit], q[bitIdx]);
    }
  }
  H(q[scratchQubit]);
  if (Measure(q[scratchQubit])) {
    out_result = 1;
    // Reset scratchQubit as well
    X(q[scratchQubit]);
  } else {
    out_result = 0;
  }
}
} // namespace ftqc
#endif