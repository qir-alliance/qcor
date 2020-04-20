#ifndef RUNTIME_QCOR_HPP_
#define RUNTIME_QCOR_HPP_

#include <memory>
#include <qalloc>

#include "Observable.hpp"
#include "Optimizer.hpp"

#include "xacc_internal_compiler.hpp"

namespace qcor {

using OptFunction = xacc::OptFunction;

Observable* PauliX(const std::size_t qbit_idx);


void set_verbose(bool verbose);

class ObjectiveFunction : public xacc::Identifiable {
protected:
  std::shared_ptr<xacc::Observable> observable;
  xacc::CompositeInstruction * kernel;
  xacc::internal_compiler::qreg qreg;

  virtual double operator()() = 0;

public:
  ObjectiveFunction() = default;
  virtual void initialize(std::shared_ptr<xacc::Observable> obs,
                          xacc::CompositeInstruction * qk) {
    observable = obs;
    kernel = qk;
  }

  void set_qreg(xacc::internal_compiler::qreg q) {
      qreg = q;
  }
  template <typename... ArgumentTypes>
  double operator()(ArgumentTypes... args) {
    kernel->updateRuntimeArguments(args...);
    return operator()();
  }
};


namespace __internal__ {

// Given a quantum kernel functor, create the xacc
// CompositeInstruction representation of it
template <typename QuantumKernel, typename... Args>
xacc::CompositeInstruction *kernel_as_composite_instruction(QuantumKernel &k,
                                                            Args... args) {
  // How to know how many qregs to create?
  // oh well just create one tmp one
  //   auto q = qalloc(1);
  // turn off execution
  xacc::internal_compiler::__execute = false;
  // Execute to compile, this will store and we can get it
  k(args...);
  // turn execution on
  xacc::internal_compiler::__execute = true;
  return xacc::internal_compiler::getLastCompiled();
}

double observe(xacc::CompositeInstruction * program,
             std::shared_ptr<xacc::Observable> obs,
             xacc::internal_compiler::qreg &q);
   

std::vector<std::shared_ptr<xacc::CompositeInstruction>>
observe(std::shared_ptr<xacc::Observable> obs,
        xacc::CompositeInstruction *program);

std::shared_ptr<ObjectiveFunction> get_objective(const char * type);


} // namespace __internal__

template <typename QuantumKernel, typename... Args>
auto observe(QuantumKernel &kernel, std::shared_ptr<xacc::Observable> obs,
             Args... args) {
  auto program = __internal__::kernel_as_composite_instruction(kernel, args...);
  return [program, obs](Args... args) {
      
    // Get the first argument, which should be a qreg
    auto q = std::get<0>(std::forward_as_tuple(args...));
    // std::cout << "\n" << program->toString() << "\n";

    // Set the arguments on the IR
    program->updateRuntimeArguments(args...);
    // std::cout << "\n" << program->toString() << "\n";

    // Observe the program
    auto programs = __internal__::observe(obs, program);

    std::vector<xacc::CompositeInstruction *> ptrs;
    for (auto p : programs)
      ptrs.push_back(p.get());

    xacc::internal_compiler::execute(q.results(), ptrs);

    // We want to contract q children buffer
    // exp-val-zs with obs term coeffs
    return q.weighted_sum(obs.get());
  }(args...);
}

// Get the default Optimizer
xacc::Optimizer *getOptimizer();

// Get a pauli observable from a string representation
std::shared_ptr<xacc::Observable> getObservable(const char *repr);

template <typename QuantumKernel, typename... KernelArguments>
std::shared_ptr<ObjectiveFunction>
createObjectiveFunction(const char *obj_name, QuantumKernel &kernel,
                        std::shared_ptr<xacc::Observable> observable,
                        KernelArguments... args) {
  auto obj_func = qcor::__internal__::get_objective(obj_name);
  auto q = std::get<0>(std::forward_as_tuple(args...));
  obj_func->set_qreg(q);
  auto program = __internal__::kernel_as_composite_instruction(kernel, args...);
  obj_func->initialize(observable, program);
  return obj_func;
}

} // namespace qcor


#endif
