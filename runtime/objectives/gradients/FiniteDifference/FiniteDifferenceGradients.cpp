/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "AlgorithmGradientStrategy.hpp"
#include "cppmicroservices/ServiceProperties.h"
#include "objective_function.hpp"
#include "gradient_function.hpp"
#include "qcor.hpp"
#include "qcor_observable.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_plugin.hpp"
#include "xacc_service.hpp"

using namespace cppmicroservices;

namespace {
// Wrapper to call XACC numerical AlgorithmGradientStrategy:
// TODO: implement QCOR native methods
std::vector<double> run_gradient_strategy(
    const std::vector<double> &x, double cost_val, const std::string &name,
    double step, qcor::Operator &obs,
    std::function<
        std::shared_ptr<qcor::CompositeInstruction>(std::vector<double>)>
        _kernel_eval) {
  auto observable =
      std::dynamic_pointer_cast<xacc::Observable>(obs.get_as_opaque());
  std::vector<double> gradients(x.size(), 0.0);
  auto gradient_strategy =
      xacc::getService<xacc::AlgorithmGradientStrategy>(name);
  if (gradient_strategy->isNumerical() && observable->getIdentitySubTerm()) {
    gradient_strategy->setFunctionValue(
        cost_val - std::real(observable->getIdentitySubTerm()->coefficient()));
  }
  std::function<std::shared_ptr<xacc::CompositeInstruction>(
      std::vector<double>)>
      kernel_eval = [&](std::vector<double> x)
      -> std::shared_ptr<xacc::CompositeInstruction> {
    // _kernel_eval produces a qcor::CompositeInstruction ptr
    return _kernel_eval(x)->as_xacc();
  };

  auto kernel = kernel_eval(x);
  gradient_strategy->initialize({{"observable", observable},
                                 {"step", step},
                                 {"kernel-evaluator", kernel_eval}});
  auto grad_kernels = gradient_strategy->getGradientExecutions(kernel, x);
  const size_t nb_qubits = std::max(static_cast<size_t>(observable->nBits()),
                                    kernel->nPhysicalBits());
  auto tmp_grad = qalloc(nb_qubits);

  xacc::internal_compiler::execute(tmp_grad.results(), grad_kernels);
  auto tmp_grad_children = tmp_grad.results()->getChildren();
  gradient_strategy->compute(gradients, tmp_grad_children);
  return gradients;
}
}  // namespace

namespace qcor {
class KernelForwardDifferenceGradient : public KernelGradientService {
 protected:
  std::shared_ptr<ObjectiveFunction> m_objFunc;
  double m_step = 1e-3;
  std::function<std::shared_ptr<qcor::CompositeInstruction>(
      std::vector<double>)>
      m_kernel_eval;

 public:
  const std::string name() const override { return "forward"; }
  const std::string description() const override { return ""; }

  virtual void initialize(
      std::function<
          std::shared_ptr<qcor::CompositeInstruction>(std::vector<double>)>
          _kernel_eval,
      Operator &obs, xacc::HeterogeneousMap &&options = {}) override {
    if (options.keyExists<double>("step")) {
      m_step = options.get<double>("step");
    }
    m_kernel_eval = _kernel_eval;
    gradient_func = [&](const std::vector<double> &x,
                        double cost_val) -> std::vector<double> {
      return run_gradient_strategy(x, cost_val, "forward", m_step, obs,
                                   m_kernel_eval);
    };
  }

  void initialize(std::shared_ptr<ObjectiveFunction> obj_func,
                  HeterogeneousMap &&options) override {
    m_objFunc = obj_func;
    if (options.keyExists<double>("step")) {
      m_step = options.get<double>("step");
    }
    gradient_func = [&](const std::vector<double> &x,
                        double cost_val) -> std::vector<double> {
      auto obs = m_objFunc->get_observable();
      return run_gradient_strategy(x, cost_val, "forward", m_step, obs,
                                   m_objFunc->get_kernel_evaluator());
    };
  }
};

class KernelBackwardDifferenceGradient : public KernelGradientService {
 protected:
  std::shared_ptr<ObjectiveFunction> m_objFunc;
  double m_step = 1e-3;
  std::function<std::shared_ptr<qcor::CompositeInstruction>(
      std::vector<double>)>
      m_kernel_eval;

 public:
  const std::string name() const override { return "backward"; }
  const std::string description() const override { return ""; }

  virtual void initialize(
      std::function<std::shared_ptr<CompositeInstruction>(std::vector<double>)>
          kernel_eval,
      Operator &obs, xacc::HeterogeneousMap &&options = {}) override {
    if (options.keyExists<double>("step")) {
      m_step = options.get<double>("step");
    }
    m_kernel_eval = kernel_eval;
    gradient_func = [&](const std::vector<double> &x,
                        double cost_val) -> std::vector<double> {
      return run_gradient_strategy(x, cost_val, "forward", m_step, obs,
                                   m_kernel_eval);
    };
  }

  void initialize(std::shared_ptr<ObjectiveFunction> obj_func,
                  HeterogeneousMap &&options) override {
    m_objFunc = obj_func;
    if (options.keyExists<double>("step")) {
      m_step = options.get<double>("step");
    }
    gradient_func = [&](const std::vector<double> &x,
                        double cost_val) -> std::vector<double> {
      auto obs = m_objFunc->get_observable();
      return run_gradient_strategy(x, cost_val, "backward", m_step, obs,
                                   m_objFunc->get_kernel_evaluator());
    };
  }
};

class KernelCentralDifferenceGradient : public KernelGradientService {
 protected:
  std::shared_ptr<ObjectiveFunction> m_objFunc;
  double m_step = 1e-3;
  std::function<std::shared_ptr<CompositeInstruction>(std::vector<double>)>
      m_kernel_eval;
  Operator m_obs;

 public:
  const std::string name() const override { return "central"; }
  const std::string description() const override { return ""; }

  virtual void initialize(
      std::function<std::shared_ptr<CompositeInstruction>(std::vector<double>)>
          kernel_eval,
      Operator &obs, xacc::HeterogeneousMap &&options = {}) override {
    if (options.keyExists<double>("step")) {
      m_step = options.get<double>("step");
    }
    m_kernel_eval = kernel_eval;
    m_obs = obs;
    gradient_func = [&](const std::vector<double> &x,
                        double cost_val) -> std::vector<double> {
      return run_gradient_strategy(x, cost_val, "central", m_step, m_obs,
                                   m_kernel_eval);
    };
  }

  void initialize(std::shared_ptr<ObjectiveFunction> obj_func,
                  HeterogeneousMap &&options) override {
    m_objFunc = obj_func;
    if (options.keyExists<double>("step")) {
      m_step = options.get<double>("step");
    }
    gradient_func = [&](const std::vector<double> &x,
                        double cost_val) -> std::vector<double> {
      auto obs = m_objFunc->get_observable();
      return run_gradient_strategy(x, cost_val, "central", m_step, obs,
                                   m_objFunc->get_kernel_evaluator());
    };
  }
};
}  // namespace qcor
namespace {
// Register all three diff plugins
class US_ABI_LOCAL FiniteDiffActivator : public BundleActivator {
 public:
  FiniteDiffActivator() {}
  void Start(BundleContext context) {
    context.RegisterService<qcor::KernelGradientService>(
        std::make_shared<qcor::KernelForwardDifferenceGradient>());
    context.RegisterService<qcor::KernelGradientService>(
        std::make_shared<qcor::KernelBackwardDifferenceGradient>());
    context.RegisterService<qcor::KernelGradientService>(
        std::make_shared<qcor::KernelCentralDifferenceGradient>());
  }
  void Stop(BundleContext /*context*/) {}
};
CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(FiniteDiffActivator)
}  // namespace
