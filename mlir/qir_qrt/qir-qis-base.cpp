#include "qir-qrt.hpp"
#include "qcor_config.hpp"
#include "qrt.hpp"
#include "xacc.hpp"
#include "xacc_config.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_service.hpp"

// Base ORNL QRT QIS (quantum instructions) implementation
extern "C" {
void __quantum__qis__cnot(Qubit *src, Qubit *tgt) {
  std::size_t src_copy = src->id;
  std::size_t tgt_copy = tgt->id;
  if (verbose)
    printf("[qir-qrt] Applying CX %lu, %lu\n", src_copy, tgt_copy);
  ::quantum::cnot({"q", src_copy}, {"q", tgt_copy});
}

void __quantum__qis__swap(Qubit *src, Qubit *tgt) {
  std::size_t src_copy = src->id;
  std::size_t tgt_copy = tgt->id;
  if (verbose)
    printf("[qir-qrt] Applying Swap %lu, %lu\n", src_copy, tgt_copy);
  ::quantum::swap({"q", src_copy}, {"q", tgt_copy});
}

void __quantum__qis__cphase(double x, Qubit *src, Qubit *tgt) {
  std::size_t src_copy = src->id;
  std::size_t tgt_copy = tgt->id;
  if (verbose)
    printf("[qir-qrt] Applying CPhase %lu, %lu\n", src_copy, tgt_copy);
  ::quantum::cphase({"q", src_copy}, {"q", tgt_copy}, x);
}

void __quantum__qis__h(Qubit *q) {
  std::size_t qcopy = q->id;
  if (verbose)
    printf("[qir-qrt] Applying H %lu\n", qcopy);
  ::quantum::h({"q", qcopy});
}

void __quantum__qis__s(Qubit *q) {
  std::size_t qcopy = q->id;
  if (verbose)
    printf("[qir-qrt] Applying S %lu\n", qcopy);
  ::quantum::s({"q", qcopy});
}

void __quantum__qis__sdg(Qubit *q) {
  std::size_t qcopy = q->id;
  if (verbose)
    printf("[qir-qrt] Applying Sdg %lu\n", qcopy);
  ::quantum::sdg({"q", qcopy});
}
void __quantum__qis__t(Qubit *q) {
  std::size_t qcopy = q->id;
  if (verbose)
    printf("[qir-qrt] Applying T %lu\n", qcopy);
  ::quantum::t({"q", qcopy});
}
void __quantum__qis__tdg(Qubit *q) {
  std::size_t qcopy = q->id;
  if (verbose)
    printf("[qir-qrt] Applying Tdg %lu\n", qcopy);
  ::quantum::tdg({"q", qcopy});
}

void __quantum__qis__x(Qubit *q) {
  std::size_t qcopy = q->id;
  if (verbose)
    printf("[qir-qrt] Applying X %lu\n", qcopy);
  ::quantum::x({"q", qcopy});
}
void __quantum__qis__y(Qubit *q) {
  std::size_t qcopy = q->id;
  if (verbose)
    printf("[qir-qrt] Applying Y %lu\n", qcopy);
  ::quantum::y({"q", qcopy});
}
void __quantum__qis__z(Qubit *q) {
  std::size_t qcopy = q->id;
  if (verbose)
    printf("[qir-qrt] Applying Z %lu\n", qcopy);
  ::quantum::z({"q", qcopy});
}

void __quantum__qis__reset(Qubit *q) {
  std::size_t qcopy = q->id;
  if (verbose)
    printf("[qir-qrt] Applying Reset %lu\n", qcopy);
  ::quantum::reset({"q", qcopy});
}

void __quantum__qis__rx(double x, Qubit *q) {
  std::size_t qcopy = q->id;
  if (verbose)
    printf("[qir-qrt] Applying Rx(%f) %lu\n", x, qcopy);
  ::quantum::rx({"q", qcopy}, x);
}

void __quantum__qis__ry(double x, Qubit *q) {
  std::size_t qcopy = q->id;
  if (verbose)
    printf("[qir-qrt] Applying Ry(%f) %lu\n", x, qcopy);
  ::quantum::ry({"q", qcopy}, x);
}

void __quantum__qis__rz(double x, Qubit *q) {
  std::size_t qcopy = q->id;
  if (verbose)
    printf("[qir-qrt] Applying Rz(%f) %lu\n", x, qcopy);
  ::quantum::rz({"q", qcopy}, x);
}
void __quantum__qis__u3(double theta, double phi, double lambda, Qubit *q) {
  std::size_t qcopy = q->id;
  if (verbose)
    printf("[qir-qrt] Applying U3(%f, %f, %f) %lu\n", theta, phi, lambda,
           qcopy);
  ::quantum::u3({"q", qcopy}, theta, phi, lambda);
}

Result *__quantum__qis__mz(Qubit *q) {
  if (verbose)
    printf("[qir-qrt] Measuring qubit %lu\n", q->id);
  std::size_t qcopy = q->id;
  auto bit = ::quantum::mz({"q", qcopy});
  if (mode == QRT_MODE::FTQC)
    if (verbose)
      printf("[qir-qrt] Result was %d.\n", bit);
  return bit ? ResultOne : ResultZero;
}
}