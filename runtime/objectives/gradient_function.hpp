#pragma once
#include "Identifiable.hpp"
#include "heterogeneous.hpp"
#include "qcor_ir.hpp"
#include "qcor_observable.hpp"
#include <functional>
#include <memory>
#include <vector>

namespace qcor {
class ObjectiveFunction;
// Gradient function type:
// Input: set of current parameters (std::vector<double>) and the current
// objective (cost) function value. Output: gradients (std::vector<double>)
// Requirements: size(parameters) == size (gradients)
using GradientFunctionType =
    std::function<std::vector<double>(const std::vector<double> &, double)>;
class GradientFunction {
protected:
  GradientFunctionType gradient_func;

public:
  GradientFunction() {}
  GradientFunction(GradientFunctionType func) : gradient_func(func) {}
  std::vector<double> operator()(const std::vector<double> &x,
                                 double current_val) {
    return gradient_func(x, current_val);
  }
};

namespace __internal__ {
extern std::string DEFAULT_GRADIENT_METHOD;
std::shared_ptr<GradientFunction>
get_gradient_method(const std::string &type,
                    std::shared_ptr<ObjectiveFunction> obj_func,
                    xacc::HeterogeneousMap options = {});

std::shared_ptr<GradientFunction>
get_gradient_method(const std::string &type,
                    std::function<std::shared_ptr<CompositeInstruction>(
                        std::vector<double>)>
                        kernel_eval,
                    Operator &obs);
} // namespace __internal__

// Interface for gradient calculation services.
// Note: we keep the base GradientFunction API as simple as possible (just a
// thin wrapper around std::function, i.e. C++ lambda) so that users can define
// it in-place if need be. We also provide a set of registered gradient
// services implementing this interface.
class KernelGradientService : public GradientFunction,
                              public xacc::Identifiable {
public:
  virtual void initialize(std::shared_ptr<ObjectiveFunction> obj_func,
                          xacc::HeterogeneousMap &&options = {}) = 0;
  virtual void
  initialize(std::function<std::shared_ptr<CompositeInstruction>(
                 std::vector<double>)>
                 kernel_eval,
             Operator &obs, xacc::HeterogeneousMap &&options = {}) = 0;
};
} // namespace qcor