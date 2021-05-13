#pragma once
#include "qcor.hpp"

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