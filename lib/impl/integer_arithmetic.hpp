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