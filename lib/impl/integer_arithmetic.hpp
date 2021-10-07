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
#include <qcor_qft>

namespace qcor {
namespace internal {
// Generates a list of angles to perform addition by a in the Fourier space.
void genAngles(std::vector<double> &io_angles, int a, int nbQubits) {
  // Makes sure the vector appropriately sized
  io_angles.resize(nbQubits);
  std::fill(io_angles.begin(), io_angles.end(), 0.0);

  for (int i = 0; i < nbQubits; ++i) {
    int bitIdx = nbQubits - i - 1;
    auto &angle = io_angles[bitIdx];
    for (int j = i; j < nbQubits; ++j) {
      // Check bit
      int bitShift = nbQubits - j - 1;
      if (((1 << bitShift) & a) != 0) {
        angle += std::pow(2.0, -(j - i));
      }
    }
    angle *= M_PI;
  }
}

inline void calcPowMultMod(int &result, int i, int a, int N) {
  result = (1 << i) * a % N;
}

// Compute extended greatest common divisor of two integers
inline std::tuple<int, int, int> egcd(int a, int b) {
  int m, n, c, q, r;
  int m1 = 0, m2 = 1, n1 = 1, n2 = 0;

  while (b) {
    q = a / b, r = a - q * b;
    m = m2 - q * m1, n = n2 - q * n1;
    a = b, b = r;
    m2 = m1, m1 = m, n2 = n1, n1 = n;
  }

  c = a, m = m2, n = n2;

  // correct the signs
  if (c < 0) {
    m = -m;
    n = -n;
    c = -c;
  }

  return std::make_tuple(m, n, c);
}

// Modular inverse of a mod p
inline void modinv(int &result, int a, int p) {
  int x, y;
  int gcd_ap;
  std::tie(x, y, gcd_ap) = egcd(p, a);
  result = (y > 0) ? y : y + p;
}
} // namespace internal
} // namespace qcor

// Majority gate
__qpu__ void majority(qubit a, qubit b, qubit c) {
  X::ctrl(c, b);
  X::ctrl(c, a);
  X::ctrl({b, a}, c);
}

// Note: this is not the adjoint of majority....
// Could be a good use case for per-qubit uncompute...
__qpu__ void unmajority(qubit a, qubit b, qubit c) {
  X::ctrl({b, a}, c);
  X::ctrl(c, a);
  X::ctrl(a, b);
}

// Add a to b and save to b
// c_in and c_out are carry bits (in and out)
__qpu__ void ripple_add(qreg a, qreg b, qubit c_in, qubit c_out) {
  majority(c_in, b[0], a[0]);
  for (auto j : range(a.size() - 1)) {
    majority(a[j], b[j + 1], a[j + 1]);
  }

  X::ctrl(a.tail(), c_out);

  for (int j = a.size() - 2; j >= 0; --j) {
    unmajority(a[j], b[j + 1], a[j + 1]);
  }
  unmajority(c_in, b[0], a[0]);
}

// Init a qubit register in an integer value
__qpu__ void integer_init(qreg a, int val) {
  Reset(a);
  for (auto i : range(a.size())) {
    if (val & (1 << i)) {
      X(a[i]);
    }
  }
}

// Add an integer to the phase (Fourier basis)
__qpu__ void phase_add_integer(qreg q, int a) {
  std::vector<double> angles;
  qcor::internal::genAngles(angles, a, q.size());
  for (int i = 0; i < q.size(); ++i) {
    U1(q[i], angles[i]);
  }
}

// Add an integer to the a qubit register:
// |a> --> |a + b>
__qpu__ void add_integer(qreg q, int a) {
  // Bring it to Fourier basis
  // (IQFT automatically)
  compute { qft_opt_swap(q, 0); }
  action { phase_add_integer(q, a); }
}

// + a mod N in phase space
// need an additional ancilla qubit since we don't do allocation
// inside a kernel.
// See Fig. 5 of https://arxiv.org/pdf/quant-ph/0205095.pdf
__qpu__ void phase_add_integer_mod(qreg q, qubit anc, int a, int N) {
  Reset(anc);
  phase_add_integer(q, a);
  phase_add_integer::adjoint(q, N);
  qft_opt_swap::adjoint(q, 0);
  X::ctrl(q.tail(), anc);
  qft_opt_swap(q, 0);
  phase_add_integer::ctrl(anc, q, N);
  phase_add_integer::adjoint(q, a);
  qft_opt_swap::adjoint(q, 0);
  X(q.tail());
  X::ctrl(q.tail(), anc);
  X(q.tail());
  qft_opt_swap(q, 0);
  phase_add_integer(q, a);
}

// Add an integer a mod N to the a qubit register:
// |q> --> |q + a mod N>
// TODO: ability to create scratch qubits
__qpu__ void add_integer_mod_impl(qreg q, qubit anc, int a, int N) {
  // Bring it to Fourier basis
  // (IQFT automatically)
  compute { qft_opt_swap(q, 0); }
  action { phase_add_integer_mod(q, anc, a, N); }
}

__qpu__ void add_integer_mod(qreg q, int a, int N) {
  auto anc = qalloc(1);
  add_integer_mod_impl(q, anc[0], a, N);
}

// Modular multiply in phase basis:
// See Fig. 6 of https://arxiv.org/pdf/quant-ph/0205095.pdf  
// |x>|b> ==> |x> |b + ax mod N>
// i.e. if b == 0 ==> the result |ax mod N> is stored in b 
__qpu__ void phase_mul_integer_mod(qreg x, qreg b, qubit anc, int a, int N) {
  for (int i = 0; i < x.size(); ++i) {
    // add operand = 2^i * a
    int operand;
    qcor::internal::calcPowMultMod(operand, i, a, N);
    phase_add_integer_mod::ctrl(x[i], b, anc, operand, N);
  }
}

__qpu__ void mul_integer_mod_impl(qreg x, qreg b, qubit anc, int a, int N) {
  // Bring it to Fourier basis
  // (IQFT automatically)
  compute { qft_opt_swap(b, 0); }
  action { phase_mul_integer_mod(x, b, anc, a, N); }
}

__qpu__ void mul_integer_mod(qreg x, qreg b, int a, int N) {
  auto anc = qalloc(1);
  mul_integer_mod_impl(x, b, anc[0], a, N);
}

// Modular multiply in-place:
// See Fig. 7 (https://arxiv.org/pdf/quant-ph/0205095.pdf)
// |x>|0> ==> |ax mod N>|0>
// x and aux_reg must have the same size.
__qpu__ void mul_integer_mod_in_place_impl(qreg x, qreg aux_reg, qubit anc, int a, int N) {
  Reset(aux_reg);
  mul_integer_mod_impl(x, aux_reg, anc, a, N);
  // Swap the result to x register
  for (int i = 0; i < x.size(); ++i) {
    Swap(x[i], aux_reg[i]);
  }

  int aInv = 0;
  qcor::internal::modinv(aInv, a, N);
  // Apply modular multiply of 1/a
  mul_integer_mod_impl::adjoint(x, aux_reg, anc, aInv, N);
}

// |x> ==> |ax mod N> in-place
__qpu__ void mul_integer_mod_in_place(qreg x, int a, int N) {
  auto anc_reg = qalloc(1);
  mul_integer_mod_in_place_impl(x, qalloc(x.size() + 1), anc_reg[0], a, N);
}