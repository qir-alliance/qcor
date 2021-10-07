/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#pragma once

// Common QCOR kernels to support iterative Quantum Phase Estimation algorithm.

// First-order Trotter evolution
__qpu__ void pauli_trotter_evolution(qreg q, qcor::PauliOperator pauli_ham,
                                     double evo_time, int num_time_slices) {
  const double theta = evo_time / num_time_slices;
  for (int i = 0; i < num_time_slices; ++i) {
    exp_i_theta(q, theta, pauli_ham);
  }
}

// Raise Pauli evolution to a power: i.e. U^power.
__qpu__ void pauli_power_trotter_evolution(qreg q,
                                           qcor::PauliOperator pauli_ham,
                                           double evo_time, int num_time_slices,
                                           int power) {
  for (int i = 0; i < power; ++i) {
    pauli_trotter_evolution(q, pauli_ham, evo_time, num_time_slices);
  }
}

// Kernel for a single iteration of iterative QPE
/// k: the iteration idx.
/// omega: the feedback angle.
/// pauli_ham: hamiltonian
/// num_time_slices: number of trotter steps
/// measure: if true (!= 0), add measure gate on ancilla qubit.
__qpu__ void pauli_qpe_iter(qreg q, int k, double omega,
                            qcor::PauliOperator pauli_ham, int num_time_slices,
                            int measure) {
  // Ancilla qubit is the last qubit in the register
  auto anc_idx = q.size() - 1;
  // Hadamard on ancilla qubit
  H(q[anc_idx]);
  // Controlled-U
  int power = 1 << (k - 1);
  double evo_time = -2 * M_PI;
  pauli_power_trotter_evolution::ctrl(anc_idx, q, pauli_ham, evo_time,
                                      num_time_slices, power);

  // Rz on ancilla qubit
  Rz(q[anc_idx], omega);
  // Hadamard on ancilla qubit
  H(q[anc_idx]);
  if (measure) {
    Measure(q[anc_idx]);
  }
}