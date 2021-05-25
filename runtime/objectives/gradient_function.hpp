#pragma once
#include <memory>
#include <vector>
#include <functional>

namespace xacc {
class CompositeInstruction;
class Observable;
}
namespace qcor {
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

// Evaluate the Forward Difference gradients of a variational kernel.
class KernelForwardDifferenceGradient : public GradientFunction {
protected:
  std::function<std::shared_ptr<xacc::CompositeInstruction>(std::vector<double>)>
      &m_kernelEval;
  double m_step;
  std::shared_ptr<xacc::Observable> m_obs;

public:
  KernelForwardDifferenceGradient(
      std::function<std::shared_ptr<xacc::CompositeInstruction>(std::vector<double>)>
          &kernel_evaluator,
      std::shared_ptr<xacc::Observable> observable, double step_size = 1.0e-7);
};
} // namespace qcor