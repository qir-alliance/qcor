#include "qir-qrt.hpp"
#include "qir-qrt-ms-compat.hpp"
#include <iostream>

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
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  // Delegate to __quantum__qis__h
  __quantum__qis__h(q);
}
void __quantum__qis__h__ctl(Array *ctls, Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__r__body(Pauli pauli, double theta, Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__r__adj(Pauli pauli, double theta, Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__r__ctl(Array *ctls, Pauli pauli, double theta, Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__r__ctladj(Array *ctls, Pauli pauli, double theta,
                               Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__s__body(Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__s(q);
}
void __quantum__qis__s__adj(Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__s__ctl(Array *ctls, Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__s__ctladj(Array *ctls, Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__t__body(Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__t(q);
}
void __quantum__qis__t__adj(Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__t__ctl(Array *ctls, Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__t__ctladj(Array *ctls, Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__x__body(Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__x(q);
}
void __quantum__qis__x__adj(Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__x__ctl(Array *ctls, Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__x__ctladj(Array *ctls, Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__y__body(Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__y(q);
}
void __quantum__qis__y__adj(Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__y__ctl(Array *ctls, Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__y__ctladj(Array *ctls, Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__z__body(Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__z(q);
}
void __quantum__qis__z__adj(Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__z__ctl(Array *ctls, Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__z__ctladj(Array *ctls, Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__qis__rx__body(double theta, Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__rx(theta, q);
}
void __quantum__qis__ry__body(double theta, Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__ry(theta, q);
}
void __quantum__qis__rz__body(double theta, Qubit *q) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__rz(theta, q);
}
void __quantum__qis__cnot__body(Qubit *src, Qubit *tgt) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  __quantum__qis__cnot(src, tgt);
}

Result *__quantum__qis__measure__body(Array *bases, Array *qubits) {
  // TODO
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  return nullptr;
}
double __quantum__qis__intasdouble__body(int32_t intVal) {
  // TODO
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