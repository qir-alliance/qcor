#include "CommonGates.hpp"
#include "qir-qrt-ms-compat.hpp"
#include "qir-qrt.hpp"
#include "qrt.hpp"
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
  case Pauli::Pauli_I: {
    // Q# use rotation aroung I to cancel global phase
    // due to Rz and U1 differences.
    // Since Q# doesn't have native CPhase gate, we need to handle
    // this properly in order for phase estimation to work.
    // Rotation(theta) aroung I is defined as:
    // diag(exp(-i*theta/2), exp(-i*theta/2)) == Phase(-theta)*Rz(theta)
    __quantum__qis__rz(theta, q);
    std::size_t qcopy = q->id;
    ::quantum::u1({"q", qcopy}, -theta);
    break;
  }
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

void __quantum__qis__r__ctl(Array *ctls, RotationCtrlArgs *args) {
  // std::cout << "Control rotation:\n";
  // std::cout << "Angle = " << args->theta << "\n";
  // std::cout << "Pauli: " << args->pauli << "\n";
  // std::cout << "Qubit: " << args->q->id << "\n";
  std::vector<Qubit *> ctrl_qubits;
  for (int i = 0; i < ctls->size(); ++i) {
    int8_t *arrayPtr = (*ctls)[i];
    Qubit *ctrl_qubit = *(reinterpret_cast<Qubit **>(arrayPtr));
    ctrl_qubits.emplace_back(ctrl_qubit);
  }
  __quantum__rt__start_ctrl_u_region();
  __quantum__qis__r__body(args->pauli, args->theta, args->q);
  __quantum__rt__end_multi_ctrl_u_region(ctrl_qubits);
}

void __quantum__qis__r__ctladj(Array *ctls, RotationCtrlArgs *args) {
  std::vector<Qubit *> ctrl_qubits;
  for (int i = 0; i < ctls->size(); ++i) {
    int8_t *arrayPtr = (*ctls)[i];
    Qubit *ctrl_qubit = *(reinterpret_cast<Qubit **>(arrayPtr));
    ctrl_qubits.emplace_back(ctrl_qubit);
  }
  __quantum__rt__start_ctrl_u_region();
  __quantum__qis__r__body(args->pauli, -args->theta, args->q);
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
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  if (mode == QRT_MODE::NISQ && enable_extended_nisq) {
    if (verbose)
      std::cout << "NISQ mode If statement generation\n";
    // NOTE: We use a different pointer value (not the static ResultOne and
    // ResultZero) in NISQ mode to store the lookup key to look up classical
    // register id to construct IfStmt IR.
    if (clb_on_zero || clb_on_one) {
      assert(r != ResultOne && r != ResultZero);
      assert(nisq_result_to_creg_idx.find(r) != nisq_result_to_creg_idx.end());
      // nisq_result_to_creg_idx should have an entry for this Result *
      const auto bit_idx = nisq_result_to_creg_idx[r];
      auto current_prog = ::quantum::qrt_impl->get_current_program();
      auto ifStmt = std::make_shared<xacc::quantum::IfStmt>();
      ifStmt->setBits({bit_idx});
      xacc::InstructionParameter creg_name("qir_creg");
      ifStmt->setParameter(0, creg_name);
      // Set the NISQ program to the ifStmt
      ::quantum::qrt_impl->set_current_program(
          std::make_shared<qcor::CompositeInstruction>(ifStmt));
      // We don't support else block atm yet.
      assert(!clb_on_zero);
      // Execute the callable: this will append NISQ instructions to the IfStmt
      // Important: implicit in this is the fact that the Callable capture the
      // whole context of the parent scope..
      clb_on_one->invoke(nullptr, nullptr);

      // Add the whole IfStmt to the program
      current_prog->addInstruction(ifStmt);
      // Restore the main program.
      ::quantum::qrt_impl->set_current_program(current_prog);
    }
  } else {
    assert(r);
    if (*r) {
      if (clb_on_one) {
        clb_on_one->invoke(nullptr, nullptr);
      }
    } else {
      if (clb_on_zero) {
        clb_on_zero->invoke(nullptr, nullptr);
      }
    }
  }
}
void __quantum__qis__applyconditionallyintrinsic__body(
    Array *rs1, Array *rs2, Callable *clb_on_equal,
    Callable *clb_on_different) {
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
}