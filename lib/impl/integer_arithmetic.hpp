#pragma once
#include "qcor.hpp"

// Majority gate
__qpu__ void majority(qubit a, qubit b, qubit c) {
  X::ctrl(c, b);
  X::ctrl(c, a);
  X::ctrl({a, b}, c);
}

// Add a to b and save to b
// c_in and c_out are carry bits (in and out)
__qpu__ void ripple_add(qreg a, qreg b, qubit c_in, qubit c_out) {
  compute {
    majority(c_in, b[0], a[0]);
    for (auto j : range(a.size() - 1)) {
      majority(a[j], b[j + 1], a[j + 1]);
    }
  }
  action { X::ctrl(a.tail(), c_out); }
}