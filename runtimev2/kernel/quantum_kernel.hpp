#pragma once

#include "qcor_utils.hpp"

namespace qcor {

template <typename Derived, typename... Args> class QuantumKernel {
protected:
  // Tuple holder for variadic kernel arguments
  std::tuple<Args...> args_tuple;

  // Parent kernel - null if this is the top-level kernel
  // not null if this is a nested kernel call
  std::shared_ptr<qcor::CompositeInstruction> parent_kernel;

  // Default, submit this kernel, if parent is given
  // turn this to false
  bool is_callable = true;

  bool disable_destructor = false;

public:
  // Default constructor, takes quantum kernel function arguments
  QuantumKernel(Args... args) : args_tuple(std::make_tuple(args...)) {}

  // Internal constructor, provide parent kernel, this
  // kernel now represents a nested kernel call and
  // appends to the parent kernel
  QuantumKernel(std::shared_ptr<qcor::CompositeInstruction> _parent_kernel,
                Args... args)
      : args_tuple(std::make_tuple(args...)), parent_kernel(_parent_kernel),
        is_callable(false) {}

  QuantumKernel() : is_callable(false) {}

  static void adjoint(std::shared_ptr<CompositeInstruction> parent_kernel,
                      Args... args) {

    // instantiate and don't let it call the destructor
    Derived derived;
    derived.disable_destructor = true;

    // run the operator()(args...) call to get the parent_kernel
    derived(args...);

    // get the instructions
    auto instructions = derived.parent_kernel->getInstructions();

    // Assert that we don't have measurement
    if (!std::all_of(
            instructions.cbegin(), instructions.cend(),
            [](const auto &inst) { return inst->name() != "Measure"; })) {
      error(
          "Unable to create Adjoint for kernels that have Measure operations.");
    }

    std::reverse(instructions.begin(), instructions.end());
    for (const auto &inst : instructions) {
      // Parametric gates:
      if (inst->name() == "Rx" || inst->name() == "Ry" ||
          inst->name() == "Rz" || inst->name() == "CPHASE") {
        inst->setParameter(0, -inst->getParameter(0).template as<double>());
      }
      // TODO: Handles T and S gates, etc... => T -> Tdg
    }

    // add the instructions to the current parent kernel
    parent_kernel->addInstructions(instructions);

    // no measures, so no execute
  }

  virtual ~QuantumKernel() {}
};
} // namespace qcor