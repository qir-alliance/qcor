/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#pragma once

#include <qcor_common>
#include <vector>

#include "qcor_observable.hpp"

#ifdef _QCOR_FTQC_RUNTIME
namespace ftqc {
__qpu__ void estimate_term_expectation(
    qreg q, const std::function<void(qreg)> &statePrep,
    std::vector<qcor::Operator> bases, int nSamples, double &out_energy) {
  double sum = 0.0;
  for (int i = 0; i < nSamples; ++i) {
    statePrep(q);
    int parity = 0;
    measure_basis(q, bases, parity);
    if (parity == 1) {
      sum = sum - 1.0;
    } else {
      sum = sum + 1.0;
    }
    reset_all(q);
  }
  out_energy = sum / nSamples;
}

// Estimates the energy of a Pauli observable by summing the energy contributed
// by the individual terms. Input:
// - observable
// The Pauli Hamiltonian.
// - nSamples
// The number of samples to use for the estimation of the term expectations.
//
// Output
// The estimated energy of the observable
__qpu__ void estimate_energy(qreg q, const std::function<void(qreg)> &statePrep,
                             qcor::Operator observable, int nSamples,
                             double &out_energy) {
  std::complex<double> energy = observable.hasIdentitySubTerm() ? observable.getIdentitySubTerm().coefficient().real() : 0.0;
  for (auto pauliInst : observable.getNonIdentitySubTerms()) {
    auto coeff = pauliInst.coefficient().real();
    auto [zv, xv] = pauliInst.toBinaryVectors(q.size());
    std::vector<qcor::Operator> ops;
    for (auto [i, x_val] : enumerate(xv)) {
      auto z_val = zv[i];
      if (x_val == z_val) {
        // Y(q[i]);
        ops.emplace_back(qcor::Y(i));
      } else if (x_val == 0) {
        // Z(q[i]);
        ops.emplace_back(qcor::Z(i));

      } else {
        // X(q[i]);
        ops.emplace_back(qcor::X(i));
      }
    }
    if (!ops.empty()) {
      double termEnergy = 0.0;
      estimate_term_expectation(q, statePrep, ops, nSamples, termEnergy);
      energy = energy + (coeff * termEnergy);
    } else {
      // Identity term:
      energy = energy + coeff;
    }
  }
  out_energy = energy.real();
}
}  // namespace ftqc
#endif
