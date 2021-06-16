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
    return q.results()->getExpectationValueZ();
  }
 
  // We have counts, so use that
  double count1 = (double)q.counts().find("1")->second;
  double count2 = (double)q.counts().find("0")->second;
  return std::fabs((count1 - count2) / (count1 + count2));
}
}  // namespace qcor
