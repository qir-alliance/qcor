#pragma once

#include <vector>
#include "qcor_observable.hpp"

// Util to reset all qubits in FTQC mode:
// TODO: we can add this as a *native* method so that the
// simulator can just do a wavefunction reset.
__qpu__ void ResetAll(qreg q) {
  for (int i = 0; i < q.size(); ++i) {
    if (Measure(q[i])) {
      X(q[i]);
    }
  }
}

// Measure tensor product of Paulis operators.
// Note: the bases must be elementary (I, X, Y, Z) Pauli ops
__qpu__ void MeasureP(qreg q, std::vector<qcor::PauliOperator> bases, int& out_parity) {
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

__qpu__ void EstimateTermExpectation(qreg q, const std::function<void(qreg)>& statePrep, std::vector<qcor::PauliOperator> bases, int nSamples, double& out_energy) {
  double sum = 0.0;
  for (int i = 0; i < nSamples; ++i) {
    statePrep(q);
    int parity = 0;
    MeasureP(q, bases, parity);
    if (parity == 1) {
      sum = sum - 1.0;
    } else {
      sum = sum + 1.0;
    }
    ResetAll(q);
  }
  out_energy = sum / nSamples;
}

// Estimates the energy of a Pauli observable by summing the energy contributed by the individual terms.
// Input:
// - observable
// The Pauli Hamiltonian.
// - nSamples
// The number of samples to use for the estimation of the term expectations.
//
// Output
// The estimated energy of the observable
__qpu__ void EstimateEnergy(qreg q, const std::function<void(qreg)>& statePrep, qcor::PauliOperator observable, int nSamples, double& out_energy) {
  std::complex<double> energy = 0.0;
  for (auto &[termStr, pauliInst] : observable.getTerms()) {
    auto coeff = pauliInst.coeff();
    auto termsMap = pauliInst.ops();
    std::vector<qcor::PauliOperator> ops;
    for (auto &[bitIdx, pauliOpStr] : termsMap) {
      if (pauliOpStr == "X") {
        ops.emplace_back(qcor::X(bitIdx));
      } 
      if (pauliOpStr == "Y") {
        ops.emplace_back(qcor::Y(bitIdx));
      } 
      if (pauliOpStr == "Z") {
        ops.emplace_back(qcor::Z(bitIdx));
      } 
    }
    if (!ops.empty()) {
      double termEnergy = 0.0;
      EstimateTermExpectation(q, statePrep, ops, nSamples, termEnergy);
      energy = energy + (coeff * termEnergy);
    }
    else {
      // Identity term:
      energy = energy + coeff;
    }
  }
  out_energy = energy.real();
}

