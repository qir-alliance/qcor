#pragma once

// #include "Optimizer.hpp"
// #include "qcor_utils.hpp"

#include <functional>
#include <memory>
#include <vector>

#include "Identifiable.hpp"
#include "heterogeneous.hpp"
#include "qcor_pimpl.hpp"

namespace qcor {

class ObjectiveFunction;

class Optimizer {
 private:
  class OptimizerImpl;
  qcor_pimpl<OptimizerImpl> m_internal;

 public:
  Optimizer();
  Optimizer(std::shared_ptr<xacc::Identifiable> generic_obj);
  OptimizerImpl *operator->();
  std::pair<double, std::vector<double>> optimize(
      std::function<double(const std::vector<double> &)> opt, const int dim);
  std::pair<double, std::vector<double>> optimize(
      std::function<double(const std::vector<double> &, std::vector<double> &)>
          opt,
      const int dim);
  std::pair<double, std::vector<double>> optimize(
      std::shared_ptr<ObjectiveFunction> obj);
  std::pair<double, std::vector<double>> optimize(ObjectiveFunction *obj);
  std::pair<double, std::vector<double>> optimize(ObjectiveFunction &obj);
  std::pair<double, std::vector<double>> optimize(ObjectiveFunction &&obj) {
    return optimize(obj);
  }
  std::string name();
  ~Optimizer();
};

// Create the desired Optimizer, delegates to xacc getOptimizer
std::shared_ptr<Optimizer> createOptimizer(
    const std::string &type, xacc::HeterogeneousMap &&options = {});

}  // namespace qcor