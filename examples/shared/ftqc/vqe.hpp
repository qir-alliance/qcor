#pragma once

#include <vector>
#include "qcor_observable.hpp"

// Measure tensor product of Paulis operators.
// Note: the bases must be elementary (I, X, Y, Z) Pauli ops
__qpu__ void MeasureP(qreg q, std::vector<qcor::PauliOperator> bases) {
  for (int i = 0; i < bases.size(); ++i) {
    auto pauliOp = bases[i];
    const std::string pauliStr = pauliOp.toString().substr(6);
    std::cout << "Pauli: " << pauliStr << "\n";
    const auto bitIdx = std::stoi(pauliStr.substr(1));
    // TODO: fix XASM compiler to handle char literal
    if (pauliStr.rfind("X", 0) == 0) {
      H(q[bitIdx]);
    }
    
    if (pauliStr.rfind("Y", 0) == 0) {
      Rx(q[bitIdx], M_PI_2);
    }
    std::cout << "Measure q[" << bitIdx << "]\n";
    Measure(q[bitIdx]);
  }
}
