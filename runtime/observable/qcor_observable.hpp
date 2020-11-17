#pragma once

#include "qcor_utils.hpp"

#include "Observable.hpp"
#include "PauliOperator.hpp"
#include "FermionOperator.hpp"

namespace qcor {

// Remap xacc types to qcor ones
using Observable = xacc::Observable;
using PauliOperator = xacc::quantum::PauliOperator;
using FermionOperator = xacc::quantum::FermionOperator;

// Convenience functions for constructing Pauli operators
PauliOperator X(int idx);
PauliOperator Y(int idx);
PauliOperator Z(int idx);
PauliOperator SP(int idx);
PauliOperator SM(int idx);
PauliOperator allZs(const int nQubits);

FermionOperator adag(int idx);
FermionOperator a(int idx);

// Expose extra algebra needed for pauli operators
PauliOperator operator+(double coeff, PauliOperator &op);
PauliOperator operator+(PauliOperator &op, double coeff);
PauliOperator operator-(double coeff, PauliOperator &op);
PauliOperator operator-(PauliOperator &op, double coeff);

// Public observe function, returns expected value of Observable
template <typename... Args>
auto observe(void (*quantum_kernel_functor)(
                 std::shared_ptr<CompositeInstruction>, Args...),
             std::shared_ptr<Observable> obs, Args... args) {
  // create a temporary with name given by mem_location_qkernel
  std::stringstream name_ss;
  name_ss << "observe_qkernel";
  auto program = qcor::__internal__::create_composite(name_ss.str());

  // Run the functor, this will add
  // all quantum instructions to the parent kernel
  quantum_kernel_functor(program, args...);
  return [program, obs](Args... args) {
    // Get the first argument, which should be a qreg
    auto q = std::get<0>(std::forward_as_tuple(args...));

    // Observe the program
    auto programs = obs->observe(program);

    xacc::internal_compiler::execute(q.results(), programs);

    // We want to contract q children buffer
    // exp-val-zs with obs term coeffs
    return q.weighted_sum(obs.get());
  }(args...);
}

// Public observe function, returns expected value of Observable
template <typename... Args>
auto observe(void (*quantum_kernel_functor)(
                 std::shared_ptr<CompositeInstruction>, Args...),
             Observable &obs, Args... args) {
  // create a temporary with name given by mem_location_qkernel
  std::stringstream name_ss;
  name_ss << "observe_qkernel";
  auto program = qcor::__internal__::create_composite(name_ss.str());

  // Run the functor, this will add
  // all quantum instructions to the parent kernel
  quantum_kernel_functor(program, args...);
  return [program, &obs](Args... args) {
    // Get the first argument, which should be a qreg
    auto q = std::get<0>(std::forward_as_tuple(args...));

    // Observe the program
    auto programs = obs.observe(program);

    xacc::internal_compiler::execute(q.results(), programs);

    // We want to contract q children buffer
    // exp-val-zs with obs term coeffs
    return q.weighted_sum(&obs);
  }(args...);
}

// Observe the given kernel, and return the expected value
double observe(std::shared_ptr<CompositeInstruction> program, Observable &obs,
               xacc::internal_compiler::qreg &q);

// Observe the given kernel, and return the expected value
double observe(std::shared_ptr<CompositeInstruction> program,
               std::shared_ptr<Observable> obs,
               xacc::internal_compiler::qreg &q);
namespace __internal__ {
// Observe the kernel and return the measured kernels
std::vector<std::shared_ptr<CompositeInstruction>>
observe(std::shared_ptr<Observable> obs,
        std::shared_ptr<CompositeInstruction> program);
} // namespace __internal__

// Create an observable from a string representation
std::shared_ptr<Observable> createObservable(const std::string &repr);
std::shared_ptr<Observable> createObservable(const std::string& name, const std::string &repr);
std::shared_ptr<Observable> createObservable(const std::string &name, HeterogeneousMap&& options);
std::shared_ptr<Observable> createObservable(const std::string &name, HeterogeneousMap& options);

std::shared_ptr<Observable> createOperator(const std::string &repr);
std::shared_ptr<Observable> createOperator(const std::string& name, const std::string &repr);
std::shared_ptr<Observable> createOperator(const std::string &name, HeterogeneousMap&& options);
std::shared_ptr<Observable> createOperator(const std::string &name, HeterogeneousMap& options);

std::shared_ptr<Observable> operatorTransform(const std::string& type, qcor::Observable& op);
std::shared_ptr<Observable> operatorTransform(const std::string& type, std::shared_ptr<Observable> op);

} // namespace qcor