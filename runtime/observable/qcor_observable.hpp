/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#pragma once

#include <complex>
#include <functional>
#include <memory>
#include <vector>

#include "Identifiable.hpp"
#include "heterogeneous.hpp"
#include "operators.hpp"
#include "qcor_pimpl.hpp"
#include "qcor_utils.hpp"

namespace qcor {

using namespace tao::operators;

// This class implements the QCOR specification Operator concept.
// It can be created from a string representation, or via the provided
// algebraic API. It is intended to be polymorphic in Operator type, i.e.
// it can be a Spin (Pauli) operator, Fermion operator, etc.
// Clients can use this by value, and operate on it algebraically with othe r
// instances of the same underlying type.
class Operator
    : public commutative_ring<Operator>,
      public equality_comparable<Operator>,
      public commutative_multipliable<Operator, double>,
      public commutative_multipliable<Operator, std::complex<double>> {
 private:
  class OperatorImpl;
  qcor_pimpl<OperatorImpl> m_internal;

 public:

  // Construct from underlying type name + config options, from string-like 
  // representation (e.g. 'X0 Y1 + Y0 Y1')
  Operator(const std::string &name, xacc::HeterogeneousMap &options);
  Operator(const std::string &type, const std::string &expr);
  Operator(const OperatorImpl &&impl);
  Operator(const Operator &i);
  Operator &operator=(const Operator &);
  Operator();
  ~Operator();

  // Transform this operator using the OperatorTransform corresponding to the 
  // key type
  Operator transform(const std::string &type, xacc::HeterogeneousMap m = {});

  // Return as an opaque Identifiable type. Clients should be able 
  // to cast this to an xacc Observable
  std::shared_ptr<xacc::Identifiable> get_as_opaque();

  // Algebraic API
  Operator &operator+=(const Operator &v) noexcept;
  Operator &operator-=(const Operator &v) noexcept;
  Operator &operator*=(const Operator &v) noexcept;
  bool operator==(const Operator &v) noexcept;
  Operator &operator*=(const double v) noexcept;
  Operator &operator*=(const std::complex<double> v) noexcept;

  // Query the number of bits
  int nQubits();
  int nBits() { return nQubits(); }

  // Observe the given CompositeInstruction to produce a vector 
  // of CompositeInstructions measured according to this Operator 
  std::vector<std::shared_ptr<CompositeInstruction>> observe(
      std::shared_ptr<CompositeInstruction> function);

  // Retrieve all Operator sub-terms as individual Operators
  std::vector<Operator> getSubTerms();
  
  // Get all sub-terms, excluding the Identity sub-term
  std::vector<Operator> getNonIdentitySubTerms();

  // Get the Identity sub-term
  Operator getIdentitySubTerm();

  // True if this Operator has an identity sub-term
  bool hasIdentitySubTerm();

  // Return a string representation of this Operator
  std::string toString() const;

  // Return the coefficient on this Operator (throws an error 
  // if there are more than 1 terms in this Operator)
  std::complex<double> coefficient();

  // Produce (zv,xv) binary representation whereby for all i in (0,nqubits-1) 
  // zv[i] == xv[i] == 1 -> Y on qubit i, zv[i] == 1, xv[i] == 0 -> Z on qubit i 
  // and xv[i] == 1, zv[i] == 0 -> X on qubit i
  std::pair<std::vector<int>, std::vector<int>> toBinaryVectors(
      const int nQubits);

  // Map qubit sites to a new set of qubits
  void mapQubitSites(std::map<int, int> &siteMap);

  // Internal class describing a sparse matrix element
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

  // Generate a sparse matrix representation of this Operator
  std::vector<SparseElement> to_sparse_matrix();

  // Compute the commutator of this operator with the given one.
  Operator commutator(Operator &);
};

void __internal_exec_observer(
    xacc::AcceleratorBuffer *,
    std::vector<std::shared_ptr<CompositeInstruction>>);

// Convenience functions for constructing Pauli operators
Operator X(int idx);
Operator Y(int idx);
Operator Z(int idx);
Operator SP(int idx);
Operator SM(int idx);
Operator allZs(const int nQubits);

// Convenience functions for constructing Fermioon operators
Operator adag(int idx);
Operator a(int idx);

// // Expose extra algebra needed for pauli operators
Operator operator+(double coeff, Operator op);
Operator operator+(Operator op, double coeff);
Operator operator-(double coeff, Operator op);
Operator operator-(Operator op, double coeff);

Eigen::MatrixXcd get_dense_matrix(Operator &op);

// Observe the given kernel, and return the expected value
double observe(std::shared_ptr<CompositeInstruction> program, Operator &obs,
               xacc::internal_compiler::qreg &q);

namespace __internal__ {
// Observe the kernel and return the measured kernels
std::vector<std::shared_ptr<CompositeInstruction>> observe(
    Operator &obs, std::shared_ptr<CompositeInstruction> program);

// Keep track of all created Operators in this translation unit
extern std::map<std::size_t, Operator> cached_observables;

}  // namespace __internal__

Operator _internal_python_createObservable(const std::string &name,
                                           const std::string &repr);

// Public QCOR Specification API for Operator creation (we retain the name Observable 
// for backwards compatibility)
Operator createObservable(const std::string &repr);
Operator createObservable(const std::string &name, const std::string &repr);
Operator createOperator(const std::string &repr);
Operator createOperator(const std::string &name, const std::string &repr);
Operator createOperator(const std::string &name, HeterogeneousMap &&options);
Operator createOperator(const std::string &name, HeterogeneousMap &options);

// Transform the given Operator
Operator operatorTransform(const std::string &type, Operator &op);


}  // namespace qcor

// Print Operators to ostreams
std::ostream &operator<<(std::ostream &os, qcor::Operator const &m);
