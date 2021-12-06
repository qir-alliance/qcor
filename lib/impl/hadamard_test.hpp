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

#include "qalloc.hpp"

using Unitary = KernelSignature<qreg>;
using StatePrep = KernelSignature<qreg>;

// <U> = <state_prep| U |state_prep>
__qpu__ void __quantum_hadamard_test(qreg q, StatePrep state_prep,
                                     Unitary unitary) {
  auto test_qubit = q.head();
  auto psi = q.extract_range({1, static_cast<std::size_t>(q.size())});

  // Prepare state |psi> on qreg of size n_qubits
  state_prep(psi);

  // Create the superposition on the first qubit
  H(test_qubit);

  // perform ctrl-U
  unitary.ctrl(test_qubit, psi);

  // add the last hadamard
  H(test_qubit);

  // measure
  Measure(test_qubit);
}

namespace qcor {
// Compute <U> = <state_prep | unitary | state_prep>
double hadamard_test(StatePrep state_prep, Unitary unitary,
                     int n_state_qubits) {
  auto q = qalloc(n_state_qubits + 1);
  __quantum_hadamard_test(q, state_prep, unitary);
  // Compute <psi|U|psi>
  // First make sure we have counts, 
  // if not, grab exp-val-z key in buffer
  auto counts = q.counts();
  if (counts.empty()) {
    return q.exp_val_z();
  }
 
  // We have counts, so use that
  // P0 - P1 = <psi|U|psi>
  double count1 = (double)q.counts().find("1")->second;
  double count2 = (double)q.counts().find("0")->second;
  return (count2 - count1) / (count1 + count2);
}
}  // namespace qcor
