#pragma once

#include "qcor_utils.hpp"

#include <complex>
#include <functional>
#include <memory>
#include <vector>

#include "Identifiable.hpp"
#include "heterogeneous.hpp"
#include "operators.hpp"
#include "qcor_pimpl.hpp"


namespace qcor {

using namespace tao::operators;

class Operator
    : public commutative_ring<Operator>,
      public equality_comparable<Operator>,
      public commutative_multipliable<Operator, double>,
      public commutative_multipliable<Operator, std::complex<double>> {
 private:
  class OperatorImpl;
  qcor_pimpl<OperatorImpl> m_internal;

 public:
  Operator(const std::string &type, const std::string &expr);
  Operator(const OperatorImpl &&impl);
  Operator(const Operator &i);
  Operator& operator=(const Operator&);
  Operator();
  ~Operator();

  std::shared_ptr<xacc::Identifiable> get_as_opaque();

  Operator &operator+=(const Operator &v) noexcept;
  Operator &operator-=(const Operator &v) noexcept;
  Operator &operator*=(const Operator &v) noexcept;
  bool operator==(const Operator &v) noexcept;
  Operator &operator*=(const double v) noexcept;
  Operator &operator*=(const std::complex<double> v) noexcept;

  int nQubits();
  int nBits() { return nQubits(); }

  std::vector<std::shared_ptr<CompositeInstruction>> observe(
      std::shared_ptr<CompositeInstruction> function);

  std::vector<Operator> getSubTerms();
  std::vector<Operator> getNonIdentitySubTerms();
  bool hasIdentitySubTerm();
  
  std::string toString() const;
  Operator getIdentitySubTerm();
  std::complex<double> coefficient();

  std::pair<std::vector<int>, std::vector<int>> toBinaryVectors(const int nQubits);
  void mapQubitSites(std::map<int, int> &siteMap);

  class SparseElement
      : std::tuple<std::uint64_t, std::uint64_t, std::complex<double>> {
   public:
    SparseElement(std::uint64_t r, std::uint64_t c,
                  std::complex<double> coeff) {
      std::get<0>(*this) = r;
      std::get<1>(*this) = c;
      std::get<2>(*this) = coeff;
    }
    std::uint64_t row() { return std::get<0>(*this); }
    std::uint64_t col() { return std::get<1>(*this); }
    const std::complex<double> coeff() { return std::get<2>(*this); }
  };
  std::vector<SparseElement> to_sparse_matrix();

  Operator commutator(Operator &);
};

void __internal_exec_observer(xacc::AcceleratorBuffer*, std::vector<std::shared_ptr<CompositeInstruction>>);

// Convenience functions for constructing Pauli operators
Operator X(int idx);
Operator Y(int idx);
Operator Z(int idx);
Operator SP(int idx);
Operator SM(int idx);
Operator allZs(const int nQubits);

Operator adag(int idx);
Operator a(int idx);

// // Expose extra algebra needed for pauli operators
Operator operator+(double coeff, Operator op);
Operator operator+(Operator op, double coeff);
Operator operator-(double coeff, Operator op);
Operator operator-(Operator op, double coeff);

// Eigen::MatrixXcd get_dense_matrix(PauliOperator &op);
// Eigen::MatrixXcd get_dense_matrix(std::shared_ptr<Observable> op);

// Observe the given kernel, and return the expected value
double observe(std::shared_ptr<CompositeInstruction> program, Operator &obs,
               xacc::internal_compiler::qreg &q);

// Observe the given kernel, and return the expected value
// double observe(std::shared_ptr<CompositeInstruction> program,
//                std::shared_ptr<Observable> obs,
//                xacc::internal_compiler::qreg &q);
namespace __internal__ {
// Observe the kernel and return the measured kernels
std::vector<std::shared_ptr<CompositeInstruction>> observe(
    Operator& obs,
    std::shared_ptr<CompositeInstruction> program);

extern std::map<std::size_t, Operator> cached_observables;

}  // namespace __internal__

Operator _internal_python_createObservable(
    const std::string &name, const std::string &repr);
    
// Create an observable from a string representation
Operator createObservable(const std::string &repr);
Operator createObservable(const std::string &name,
                                             const std::string &repr);
// Operator createObservable(const std::string &name,
//                                              HeterogeneousMap &&options);
// Operator createObservable(const std::string &name,
//                                              HeterogeneousMap &options);

Operator createOperator(const std::string &repr);
Operator createOperator(const std::string &name,
                                           const std::string &repr);
// Operator createOperator(const std::string &name,
//                                            HeterogeneousMap &&options);
// Operator createOperator(const std::string &name,
//                                            HeterogeneousMap &options);

// Operator operatorTransform(const std::string &type,
//                                               Operator &op);
// Operator operatorTransform(const std::string &type,
//                                               std::shared_ptr<Observable> op);

}  // namespace qcor

std::ostream &operator<<(std::ostream &os, qcor::Operator const &m);
