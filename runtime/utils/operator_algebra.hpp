#ifndef RUNTIME_QCOR_OP_ALG_HPP_
#define RUNTIME_QCOR_OP_ALG_HPP_

#include "qalloc.hpp"
#include <CompositeInstruction.hpp>
#include <memory>

#include "FermionOperator.hpp"
#include "ObservableTransform.hpp"
#include "PauliOperator.hpp"
#include "Observable.hpp"

namespace qcor{
using PauliOperator = xacc::quantum::PauliOperator;
using FermionOperator = xacc::quantum::FermionOperator;

PauliOperator X(int idx) { return PauliOperator({{idx, "X"}}); }
PauliOperator Y(int idx) { return PauliOperator({{idx, "Y"}}); }
PauliOperator Z(int idx) { return PauliOperator({{idx, "Z"}}); }
PauliOperator SP(int idx) {
  std::complex<double> imag(0.0, 1.0);
  return X(idx) + imag * Y(idx);
}
PauliOperator SM(int idx) {
  std::complex<double> imag(0.0, 1.0);
  return X(idx) - imag * Y(idx);
}
FermionOperator a(int idx) {
  std::string s("(1.0, 0) " + std::to_string(idx));
  return FermionOperator(s);
}
FermionOperator adag(int idx) {
  std::string s("(1.0, 0) " + std::to_string(idx) + "^");
  return FermionOperator(s);
}

PauliOperator allZs(const int nQubits) {
  auto ret = Z(0);
  for (int i = 1; i < nQubits; i++) {
    ret *= Z(i);
  }
  return ret;
}

// transform FermionOperator to PauliOperator
PauliOperator transform(FermionOperator &obs, std::string transf = "jw");

template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
PauliOperator operator+(T coeff, PauliOperator &op) {
  return PauliOperator(coeff) + op;
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
PauliOperator operator+(PauliOperator &op, T coeff) {
  return PauliOperator(coeff) + op;
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
PauliOperator operator-(T coeff, PauliOperator &op) {
  return -1.0 * coeff + op;
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
PauliOperator operator-(PauliOperator &op, T coeff) {
  return -1.0 * coeff + op;
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
FermionOperator operator+(T coeff, FermionOperator &op) {
  return FermionOperator(coeff) + op;
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
FermionOperator operator+(FermionOperator &op, T coeff) {
  return FermionOperator(coeff) + op;
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
FermionOperator operator-(T coeff, FermionOperator &op) {
  return -1.0 * coeff + op;
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
FermionOperator operator-(FermionOperator &op, T coeff) {
  return -1.0 * coeff + op;
}

template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
PauliOperator operator+(T coeff, PauliOperator &&op) {
  return PauliOperator(coeff) + op;
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
PauliOperator operator+(PauliOperator &&op, T coeff) {
  return PauliOperator(coeff) + op;
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
PauliOperator operator-(T coeff, PauliOperator &&op) {
  return -1.0 * coeff + op;
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
PauliOperator operator-(PauliOperator &&op, T coeff) {
  return -1.0 * coeff + op;
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
FermionOperator operator+(T coeff, FermionOperator &&op) {
  return FermionOperator(coeff) + op;
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
FermionOperator operator+(FermionOperator &&op, T coeff) {
  return FermionOperator(coeff) + op;
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
FermionOperator operator-(T coeff, FermionOperator &&op) {
  return -1.0 * coeff + op;
}
template <typename T, typename = typename std::enable_if<
                          std::is_arithmetic<T>::value, T>::type>
FermionOperator operator-(FermionOperator &&op, T coeff) {
  return -1.0 * coeff + op;
}

PauliOperator operator+(FermionOperator &&fop, PauliOperator &&pop) {
  auto pfop = transform(fop);
  return pfop + pop;
}

PauliOperator operator+(PauliOperator &&pop, FermionOperator &&fop) {
  auto pfop = transform(fop);
  return pop + pfop;
}

PauliOperator operator*(PauliOperator &&pop, FermionOperator &&fop) {
  auto pfop = transform(fop);
  return pfop * pop;
}

PauliOperator operator*(FermionOperator &&fop, PauliOperator &&pop) {
  auto pfop = transform(fop);
  return pop * pfop;
}

PauliOperator operator+(FermionOperator &fop, PauliOperator &pop) {
  auto pfop = transform(fop);
  return pfop + pop;
}

PauliOperator operator+(PauliOperator &pop, FermionOperator &fop) {
  auto pfop = transform(fop);
  return pop + pfop;
}

PauliOperator operator*(PauliOperator &pop, FermionOperator &fop) {
  auto pfop = transform(fop);
  return pfop * pop;
}

PauliOperator operator*(FermionOperator &fop, PauliOperator &pop) {
  auto pfop = transform(fop);
  return pop * pfop;
}

}
#endif