#include "qir-qrt-ms-compat.hpp"
#include "qir-qrt.hpp"
#include <iostream>
#include <math.h>
#include <stdexcept>

namespace {
static std::vector<Pauli> extractPauliIds(Array *paulis) {
  std::vector<Pauli> pauliIds;
  for (int i = 0; i < paulis->size(); ++i) {
    pauliIds.emplace_back(static_cast<Pauli>(*paulis->getItemPointer(i)));
  }
  return pauliIds;
}
} // namespace
// QRT QIS (quantum instructions) implementation for MSFT Compatability
extern "C" {
void __quantum__qis__exp__body(Array *paulis, double angle, Array *qubits) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__exp__adj(Array *paulis, double angle, Array *qubits) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__exp__ctl(Array *ctls, Array *paulis, double angle,
                              Array *qubits) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__exp__ctladj(Array *ctls, Array *paulis, double angle,
                                 Array *qubits) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__h__body(Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  // Delegate to __quantum__qis__h
  __quantum__qis__h(q);
}
void __quantum__qis__h__ctl(Array *ctls, Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  std::vector<Qubit *> ctrl_qubits;
  for (int i = 0; i < ctls->size(); ++i) {
    int8_t *arrayPtr = (*ctls)[i];
    Qubit *ctrl_qubit = *(reinterpret_cast<Qubit **>(arrayPtr));
    ctrl_qubits.emplace_back(ctrl_qubit);
  }
  __quantum__rt__start_ctrl_u_region();
  __quantum__qis__h(q);
  __quantum__rt__end_multi_ctrl_u_region(ctrl_qubits);
}

void __quantum__qis__r__body(Pauli pauli, double theta, Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  switch (pauli) {
  case Pauli::Pauli_I:
    // nothing to do
    break;
  case Pauli::Pauli_X: {
    __quantum__qis__rx(theta, q);
    break;
  }
  case Pauli::Pauli_Y: {
    __quantum__qis__ry(theta, q);
    break;
  }
  case Pauli::Pauli_Z: {
    __quantum__qis__rz(theta, q);
    break;
  }
  default:
    __builtin_unreachable();
  }
}
void __quantum__qis__r__adj(Pauli pauli, double theta, Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__r__body(pauli, -theta, q);
}
void __quantum__qis__r__ctl(Array *ctls, Pauli pauli, double theta, Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  std::vector<Qubit *> ctrl_qubits;
  for (int i = 0; i < ctls->size(); ++i) {
    int8_t *arrayPtr = (*ctls)[i];
    Qubit *ctrl_qubit = *(reinterpret_cast<Qubit **>(arrayPtr));
    ctrl_qubits.emplace_back(ctrl_qubit);
  }
  __quantum__rt__start_ctrl_u_region();
  __quantum__qis__r__body(pauli, theta, q);
  __quantum__rt__end_multi_ctrl_u_region(ctrl_qubits);
}
void __quantum__qis__r__ctladj(Array *ctls, Pauli pauli, double theta,
                               Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  std::vector<Qubit *> ctrl_qubits;
  for (int i = 0; i < ctls->size(); ++i) {
    int8_t *arrayPtr = (*ctls)[i];
    Qubit *ctrl_qubit = *(reinterpret_cast<Qubit **>(arrayPtr));
    ctrl_qubits.emplace_back(ctrl_qubit);
  }
  __quantum__rt__start_ctrl_u_region();
  __quantum__qis__r__body(pauli, -theta, q);
  __quantum__rt__end_multi_ctrl_u_region(ctrl_qubits);
}
void __quantum__qis__s__body(Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__s(q);
}
void __quantum__qis__s__adj(Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__sdg(q);
}

void __quantum__qis__s__ctl(Array *ctls, Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  std::vector<Qubit *> ctrl_qubits;
  for (int i = 0; i < ctls->size(); ++i) {
    int8_t *arrayPtr = (*ctls)[i];
    Qubit *ctrl_qubit = *(reinterpret_cast<Qubit **>(arrayPtr));
    ctrl_qubits.emplace_back(ctrl_qubit);
  }
  __quantum__rt__start_ctrl_u_region();
  __quantum__qis__s__body(q);
  __quantum__rt__end_multi_ctrl_u_region(ctrl_qubits);
}

void __quantum__qis__s__ctladj(Array *ctls, Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  std::vector<Qubit *> ctrl_qubits;
  for (int i = 0; i < ctls->size(); ++i) {
    int8_t *arrayPtr = (*ctls)[i];
    Qubit *ctrl_qubit = *(reinterpret_cast<Qubit **>(arrayPtr));
    ctrl_qubits.emplace_back(ctrl_qubit);
  }
  __quantum__rt__start_ctrl_u_region();
  __quantum__qis__s__adj(q);
  __quantum__rt__end_multi_ctrl_u_region(ctrl_qubits);
}

void __quantum__qis__t__body(Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__t(q);
}
void __quantum__qis__t__adj(Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__tdg(q);
}
void __quantum__qis__t__ctl(Array *ctls, Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  std::vector<Qubit *> ctrl_qubits;
  for (int i = 0; i < ctls->size(); ++i) {
    int8_t *arrayPtr = (*ctls)[i];
    Qubit *ctrl_qubit = *(reinterpret_cast<Qubit **>(arrayPtr));
    ctrl_qubits.emplace_back(ctrl_qubit);
  }
  __quantum__rt__start_ctrl_u_region();
  __quantum__qis__t__body(q);
  __quantum__rt__end_multi_ctrl_u_region(ctrl_qubits);
}
void __quantum__qis__t__ctladj(Array *ctls, Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  std::vector<Qubit *> ctrl_qubits;
  for (int i = 0; i < ctls->size(); ++i) {
    int8_t *arrayPtr = (*ctls)[i];
    Qubit *ctrl_qubit = *(reinterpret_cast<Qubit **>(arrayPtr));
    ctrl_qubits.emplace_back(ctrl_qubit);
  }
  __quantum__rt__start_ctrl_u_region();
  __quantum__qis__t__adj(q);
  __quantum__rt__end_multi_ctrl_u_region(ctrl_qubits);
}

void __quantum__qis__x__body(Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__x(q);
}

void __quantum__qis__x__adj(Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  // Self-adjoint
  __quantum__qis__x__body(q);
}

void __quantum__qis__x__ctl(Array *ctls, Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  if (ctls && ctls->size() == 1) {
    int8_t *arrayPtr = (*ctls)[0];
    Qubit *ctrl_qubit = *(reinterpret_cast<Qubit **>(arrayPtr));
    __quantum__qis__cnot(ctrl_qubit, q);
  } else {
    std::vector<Qubit *> ctrl_qubits;
    for (int i = 0; i < ctls->size(); ++i) {
      int8_t *arrayPtr = (*ctls)[i];
      Qubit *ctrl_qubit = *(reinterpret_cast<Qubit **>(arrayPtr));
      ctrl_qubits.emplace_back(ctrl_qubit);
    }
    __quantum__rt__start_ctrl_u_region();
    __quantum__qis__x__body(q);
    __quantum__rt__end_multi_ctrl_u_region(ctrl_qubits);
  }
}

void __quantum__qis__x__ctladj(Array *ctls, Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  // Self-adjoint
  return __quantum__qis__x__ctl(ctls, q);
}

void __quantum__qis__y__body(Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__y(q);
}
void __quantum__qis__y__adj(Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  // Self-adjoint
  __quantum__qis__y__body(q);
}
void __quantum__qis__y__ctl(Array *ctls, Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  std::vector<Qubit *> ctrl_qubits;
  for (int i = 0; i < ctls->size(); ++i) {
    int8_t *arrayPtr = (*ctls)[i];
    Qubit *ctrl_qubit = *(reinterpret_cast<Qubit **>(arrayPtr));
    ctrl_qubits.emplace_back(ctrl_qubit);
  }
  __quantum__rt__start_ctrl_u_region();
  __quantum__qis__y__body(q);
  __quantum__rt__end_multi_ctrl_u_region(ctrl_qubits);
}
void __quantum__qis__y__ctladj(Array *ctls, Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  // Self-adjoint
  __quantum__qis__y__ctl(ctls, q);
}
void __quantum__qis__z__body(Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__z(q);
}
void __quantum__qis__z__adj(Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  // Self-adjoint
  __quantum__qis__z__body(q);
}
void __quantum__qis__z__ctl(Array *ctls, Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  std::vector<Qubit *> ctrl_qubits;
  for (int i = 0; i < ctls->size(); ++i) {
    int8_t *arrayPtr = (*ctls)[i];
    Qubit *ctrl_qubit = *(reinterpret_cast<Qubit **>(arrayPtr));
    ctrl_qubits.emplace_back(ctrl_qubit);
  }
  __quantum__rt__start_ctrl_u_region();
  __quantum__qis__z__body(q);
  __quantum__rt__end_multi_ctrl_u_region(ctrl_qubits);
}
void __quantum__qis__z__ctladj(Array *ctls, Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  // Self-adjoint
  __quantum__qis__z__ctl(ctls, q);
}
void __quantum__qis__rx__body(double theta, Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__rx(theta, q);
}
void __quantum__qis__ry__body(double theta, Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__ry(theta, q);
}
void __quantum__qis__rz__body(double theta, Qubit *q) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__rz(theta, q);
}
void __quantum__qis__cnot__body(Qubit *src, Qubit *tgt) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__cnot(src, tgt);
}

Result *__quantum__qis__measure__body(Array *bases, Array *qubits) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  if (bases->size() != qubits->size()) {
    throw "Invalid Measure instruction: the list of bases must match the list "
          "of qubits.";
  }
  const auto paulis = extractPauliIds(bases);
  std::vector<bool> results;
  for (size_t i = 0; i < paulis.size(); ++i) {
    const auto pauli = paulis[i];
    int8_t *arrayPtr = (*qubits)[i];
    Qubit *q = *(reinterpret_cast<Qubit **>(arrayPtr));
    switch (pauli) {
    case Pauli::Pauli_I:
      // std::cout << "I";
      results.emplace_back(true);
      break;
    case Pauli::Pauli_X: {
      // std::cout << "X";
      __quantum__qis__h(q);
      Result *bit_result = __quantum__qis__mz(q);
      results.emplace_back(*bit_result);
      break;
    }

    case Pauli::Pauli_Y: {
      // std::cout << "Y";
      __quantum__qis__rx(M_PI / 2.0, q);
      Result *bit_result = __quantum__qis__mz(q);
      results.emplace_back(*bit_result);
      break;
    }

    case Pauli::Pauli_Z: {
      // std::cout << "Z";
      Result *bit_result = __quantum__qis__mz(q);
      results.emplace_back(*bit_result);
      break;
    }

    default:
      __builtin_unreachable();
    }
  }
  // std::cout << "\n";
  // All equal measure result:
  if (std::equal(results.begin() + 1, results.end(), results.begin())) {
    return results[0] ? ResultOne : ResultZero;
  }
  // Zero otherwise:
  return ResultZero;
}

double __quantum__qis__intasdouble__body(int32_t intVal) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  return static_cast<double>(intVal);
}

void __quantum__qis__reset__body(Qubit *q) { __quantum__qis__reset(q); }

void __quantum__qis__applyifelseintrinsic__body(Result *r,
                                                Callable *clb_on_zero,
                                                Callable *clb_on_one) {
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__applyconditionallyintrinsic__body(
    Array *rs1, Array *rs2, Callable *clb_on_equal,
    Callable *clb_on_different) {
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
}