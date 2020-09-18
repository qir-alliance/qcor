#pragma once
#include <qalloc>
#include <vector>
#include "qcor_observable.hpp"

#ifdef _QCOR_FTQC_RUNTIME 
namespace ftqc {
__qpu__ void ResetAll(qreg q) {
  for (int i = 0; i < q.size(); ++i) {
    if (Measure(q[i])) {
      X(q[i]);
    }
  }
}

// FTQC "sync" Pauli measurement: returns the parity output
__qpu__ void MeasureP(qreg q, std::vector<qcor::PauliOperator> bases,
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
}
#endif