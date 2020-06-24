#pragma once

#include "qcor.hpp"

namespace qcor {

namespace __internal__ {
// This simple struct is a way for us to 
// enumerate commonly seen TranslationFunctors, a utility 
// that maps quantum kernel argument structure to the 
// qcor Optimizer / OptFunction std::vector<double> x parameters.
struct TranslationFunctorGenerator {

  qcor::TranslationFunctor<qreg, double> operator()(qreg &q,
                                                    std::tuple<double> &&);
  qcor::TranslationFunctor<qreg, std::vector<double>>
  operator()(qreg &q, std::tuple<std::vector<double>> &&);
};
} // namespace __internal__

// High-level VQE class, enables programmers to 
// easily construct the VQE task given an parameterized 
// qcor quantum kernel, the Hamiltonian / Observable of 
// interest, and the dimension or number of parameters 
// that make up the parameterized circuit
template <typename QuantumKernel> class VQE {
protected:

  // Reference to the paramerized 
  // quantum kernel functor
  QuantumKernel &ansatz;

  // Reference to the Hamiltonian / Observable, 
  // will dictate measurements on the kernel
  Observable &observable;

  // We need the user to tell us the number of 
  // parameters in the parameterized circuit
  std::size_t n_params;

  // Register of qubits to operate on
  qreg q;

public:
  // Typedef for describing the energy / params return type
  using VQEResultType = std::pair<double, std::vector<double>>;

  // Constructor
  VQE(QuantumKernel &kernel, Observable &obs,
      const std::size_t n_circuit_parameters)
      : ansatz(kernel), observable(obs), n_params(n_circuit_parameters) {
    q = qalloc(obs.nBits());
  }

  // Execute the VQE task synchronously, assumes default optimizer
  template <typename... Args> VQEResultType execute() {
    auto optimizer = qcor::createOptimizer("nlopt");
    auto handle = execute_async<Args...>(optimizer);
    return this->sync(handle);
  }

  // Execute the VQE task asynchronously, default optimizer
  template <typename... Args> Handle execute_async() {
    auto optimizer = qcor::createOptimizer("nlopt");
    return execute_async<Args...>(optimizer);
  }

  // Execute the VQE task synchronously, use provided Optimizer
  template <typename... Args>
  VQEResultType execute(std::shared_ptr<Optimizer> optimizer) {
    auto handle = execute_async<Args...>(optimizer);
    return this->sync(handle);
  }

  // Execute the VQE task asynchronously, use provided Optimizer
  template <typename... Args>
  Handle execute_async(std::shared_ptr<Optimizer> optimizer) {
    // Get the VQE ObjectiveFunction
    auto objective = qcor::createObjectiveFunction("vqe", ansatz, observable);

    // Create the Arg Translator
    __internal__::TranslationFunctorGenerator gen;
    auto arg_translator = gen(q, std::tuple<Args...>());

    // Run TaskInitiate, kick of the VQE job asynchronously
    return qcor::taskInitiate(objective, optimizer, arg_translator, n_params);
  }

  // Sync up the results with the host thread
  VQEResultType sync(Handle &h) {
    auto results = qcor::sync(h);
    return std::make_pair(results.opt_val, results.opt_params);
  }
};

// Next, add QAOA...

} // namespace qcor
