#pragma once

#include "qcor_utils.hpp"

#include "Observable.hpp"
#include "PauliOperator.hpp"

namespace qcor {

using Observable = xacc::Observable;
using PauliOperator = xacc::quantum::PauliOperator;

PauliOperator X(int idx);
PauliOperator Y(int idx);
PauliOperator Z(int idx);
PauliOperator SP(int idx);
PauliOperator SM(int idx);
PauliOperator allZs(const int nQubits);
 
template <typename T> PauliOperator operator+(T coeff, PauliOperator &op) {
  return PauliOperator(coeff) + op;
}
template <typename T> PauliOperator operator+(PauliOperator &op, T coeff) {
  return PauliOperator(coeff) + op;
}

template <typename T> PauliOperator operator-(T coeff, PauliOperator &op) {
  return -1.0 * coeff + op;
}

template <typename T> PauliOperator operator-(PauliOperator &op, T coeff) {
  return -1.0 * coeff + op;
}

// Public observe function, returns expected value of Observable
template <typename QuantumKernel, typename... Args>
auto observe(QuantumKernel &kernel, std::shared_ptr<Observable> obs,
             Args... args) {
  auto program = __internal__::kernel_as_composite_instruction(kernel, args...);
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

template <typename QuantumKernel, typename... Args>
auto observe(QuantumKernel &kernel, Observable &obs, Args... args) {
  auto program = __internal__::kernel_as_composite_instruction(kernel, args...);
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

// Observe the kernel and return the measured kernels
std::vector<std::shared_ptr<CompositeInstruction>>
observe(std::shared_ptr<Observable> obs,
        std::shared_ptr<CompositeInstruction> program);

// Create an observable from a string representation
std::shared_ptr<Observable> createObservable(const std::string &repr);
} // namespace qcor