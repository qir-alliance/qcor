#pragma once

#include <vector>
#include "qcor_observable.hpp"
#include <qcor_common>

__qpu__ void EstimateTermExpectation(qreg q, const std::function<void(qreg)>& statePrep, std::vector<qcor::PauliOperator> bases, int nSamples, double& out_energy) {
  double sum = 0.0;
  for (int i = 0; i < nSamples; ++i) {
    statePrep(q);
    int parity = 0;
    ftqc::measure_basis(q, bases, parity);
    if (parity == 1) {
      sum = sum - 1.0;
    } else {
      sum = sum + 1.0;
    }
    ftqc::reset_all(q);
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

