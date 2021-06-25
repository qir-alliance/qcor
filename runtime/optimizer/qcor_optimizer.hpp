#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "Identifiable.hpp"
#include "heterogeneous.hpp"
#include "qcor_pimpl.hpp"

namespace qcor {

class ObjectiveFunction;

// This class provides the QCOR specification Optimizer data type. 
// It's role is to take an ObjectiveFunction as input and execute 
// an implementation specific, user-specified, optimization workflow to 
// compute the optimal value and parameters for the provided function. 
class Optimizer {
 private:
  // Delegate the actual implementation to 
  // this hiddent opaque type. 
  class OptimizerImpl;
  qcor_pimpl<OptimizerImpl> m_internal;

 public:

  // Constructors, take the name of the 
  // concrete Optimizer implementation, an 
  // instance of an XACC optimizer, or the 
  // name plus pertinent config options.
  Optimizer();
  Optimizer(const std::string& name);
  Optimizer(const std::string& name, xacc::HeterogeneousMap&& options);
  Optimizer(std::shared_ptr<xacc::Identifiable> generic_obj);

  // Delegate to the internal implementation
  OptimizerImpl *operator->();

  // Can optimize general <functional> std functions (must provide dimensions)...
  std::pair<double, std::vector<double>> optimize(
      std::function<double(const std::vector<double> &)> opt, const int dim);
  std::pair<double, std::vector<double>> optimize(
      std::function<double(const std::vector<double> &, std::vector<double> &)>
          opt,
      const int dim);

  // or Objective function pointers, reference, or rvalue reference
  std::pair<double, std::vector<double>> optimize(
      std::shared_ptr<ObjectiveFunction> obj);
  std::pair<double, std::vector<double>> optimize(ObjectiveFunction *obj);
  std::pair<double, std::vector<double>> optimize(ObjectiveFunction &obj);
  std::pair<double, std::vector<double>> optimize(ObjectiveFunction &&obj) {
    return optimize(obj);
  }

  // Return the implementation name.
  std::string name();
  ~Optimizer();
};

// Create the desired Optimizer
std::shared_ptr<Optimizer> createOptimizer(
    const std::string &type, xacc::HeterogeneousMap &&options = {});

}  // namespace qcor