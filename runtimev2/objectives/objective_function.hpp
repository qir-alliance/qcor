#pragma once

#include "qcor_observable.hpp"
#include "qcor_utils.hpp"
#include "quantum_kernel.hpp"

namespace qcor {

class ObjectiveFunction;

namespace __internal__ {
// Get the objective function from the service registry
std::shared_ptr<ObjectiveFunction> get_objective(const std::string &type);

} // namespace __internal__

// The ObjectiveFunction represents a functor-like data structure that
// models a general parameterized scalar function. It is initialized with a
// problem-specific Observable and Quantum Kernel, and exposes a method for
// evaluation, given a list or array of scalar parameters.
// Implementations of this concept are problem-specific, and leverage the
// observe() functionality of the provided Observable to produce one or many
// measured Kernels that are then queued for execution on the available quantum
// co-processor, given the current value of the input parameters. The results of
// these quantum executions are to be used by the ObjectiveFunction to return a
// list of scalar values, representing the evaluation of the ObjectiveFunction
// at the given set of input parameters. Furthermore, the ObjectiveFunction has
// access to a global ResultBuffer that it uses to publish execution results at
// the current input parameters
class ObjectiveFunction : public xacc::Identifiable {
private:
  // This points to provided functor representation
  // of the quantum kernel, used to reconstruct
  // CompositeInstruction in variadic operator()
  void *pointer_to_functor = nullptr;

protected:
  // Pointer to the problem-specific Observable
  Observable *observable;

  // Pointer to the quantum kernel
  std::shared_ptr<CompositeInstruction> kernel;

  // The buffer containing all execution results
  xacc::internal_compiler::qreg qreg;
  bool kernel_is_xacc_composite = false;

  HeterogeneousMap options;
  std::vector<double> current_iterate_parameters;

  // To be implemented by subclasses. Subclasses
  // can assume that the kernel has been evaluated
  // at current iterates (or evaluation) of the
  // objective function. I.e. this is called in
  // the variadic operator()(Args... args) method after
  // kernel->updateRuntimeArguments(args...)
  virtual double operator()() = 0;

public:
  // Publicly visible to clients for use in Optimization
  std::vector<double> current_gradient;

  // The Constructor
  ObjectiveFunction() = default;

  // Initialize this ObjectiveFunction with the problem
  // specific observable and CompositeInstruction
  virtual void initialize(Observable *obs,
                          std::shared_ptr<CompositeInstruction> qk) {
    observable = obs;
    kernel = qk;
    kernel_is_xacc_composite = true;
  }

  // Initialize this ObjectiveFunction with the problem
  // specific observable and pointer to quantum functor
  virtual void initialize(Observable *obs, void *qk) {
    observable = obs;
    pointer_to_functor = qk;
  }

  void set_options(HeterogeneousMap &opts) { options = opts; }

  // Set the results buffer
  void set_qreg(xacc::internal_compiler::qreg &q) { qreg = q; }
  xacc::internal_compiler::qreg get_qreg() { return qreg; }

  // Evaluate this Objective function at the give parameters.
  // These variadic parameters must mirror the provided
  // quantum kernel
  template <typename... ArgumentTypes>
  double operator()(ArgumentTypes... args) {
    void (*functor)(std::shared_ptr<CompositeInstruction>, ArgumentTypes...);
    if (pointer_to_functor) {
      functor =
          reinterpret_cast<void (*)(std::shared_ptr<CompositeInstruction>,
                                    ArgumentTypes...)>(pointer_to_functor);
    }

    if (!qreg.results()) {
      // this hasn't been set, so set it
      qreg = std::get<0>(std::forward_as_tuple(args...));
    }

    if (kernel_is_xacc_composite) {
      kernel->updateRuntimeArguments(args...);
    } else {
      // create a temporary
      std::stringstream name_ss;
      name_ss << this << "_qkernel";
      kernel = qcor::__internal__::create_composite(name_ss.str());
      functor(kernel, args...);

      current_iterate_parameters.clear();
      __internal__::ConvertDoubleLikeToVectorDouble convert(
          current_iterate_parameters);
      __internal__::tuple_for_each(std::make_tuple(args...), convert);
    }
    return operator()();
  }
};

std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    const std::string &obj_name, std::shared_ptr<CompositeInstruction> kernel,
    std::shared_ptr<Observable> observable, HeterogeneousMap &&options = {});

std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    const std::string &obj_name, std::shared_ptr<CompositeInstruction> kernel,
    Observable &observable, HeterogeneousMap &&options = {});

// Create an Objective Function that makes calls to the
// provided Quantum Kernel, with measurements dictated by
// the provided Observable. Optionally can provide problem-specific
// options map.
template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    const std::string &obj_name,
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    std::shared_ptr<Observable> observable, HeterogeneousMap &&options = {}) {
  auto obj_func = qcor::__internal__::get_objective(obj_name);
  // We can store this function pointer to a void* on ObjectiveFunction
  // to be converted to CompositeInstruction later
  void *kk = reinterpret_cast<void *>(quantum_kernel_functor);
  obj_func->initialize(observable.get(), kk);
  obj_func->set_options(options);
  return obj_func;
}

// This method takes as input a functor with a signature
// that takes a CompositeInstruction as the first input argument,
// followed by other arguments that feed into the creation of the
// quantum kernel.
template <typename... Args>
std::shared_ptr<ObjectiveFunction> createObjectiveFunction(
    const std::string &obj_name,
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    Observable &observable, HeterogeneousMap &&options = {}) {
  auto obj_func = qcor::__internal__::get_objective(obj_name);
  // We can store this function pointer to a void* on ObjectiveFunction
  // to be converted to CompositeInstruction later
  void *kk = reinterpret_cast<void *>(quantum_kernel_functor);
  obj_func->initialize(&observable, kk);
  obj_func->set_options(options);
  return obj_func;
}

} // namespace qcor