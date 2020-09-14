#include "qcor.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
using namespace cppmicroservices;

#include <memory>
#include <set>

#include "AlgorithmGradientStrategy.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_service.hpp"


namespace qcor {

class VQEObjective : public ObjectiveFunction {
public:
  std::shared_ptr<xacc::Algorithm> vqe;
  double operator()(xacc::internal_compiler::qreg &qreg,
                    std::vector<double> &dx) override {
    if (!vqe) {
      vqe = xacc::getAlgorithm("vqe");
    }
    auto qpu = xacc::internal_compiler::get_qpu();
    auto success = vqe->initialize(
        {{"ansatz", kernel}, {"accelerator", qpu}, {"observable", observable}});

    if (!success) {
      xacc::error(
          "QCOR VQE Error - could not initialize internal xacc vqe algorithm.");
    }

    auto tmp_child = qalloc(qreg.size());
    auto val = vqe->execute(xacc::as_shared_ptr(tmp_child.results()), {})[0];
    // want to store parameters, have to do it here
    for (auto &child : tmp_child.results()->getChildren()) {
      child->addExtraInfo("parameters", current_iterate_parameters);
      auto tmp = current_iterate_parameters;
      tmp.push_back(val);
      child->addExtraInfo("qcor-params-energy", tmp);
    }
    qreg.addChild(tmp_child);

    if (!dx.empty() && options.stringExists("gradient-strategy")) {
      // Compute the gradient
      auto gradient_strategy =
          xacc::getService<xacc::AlgorithmGradientStrategy>(
              options.getString("gradient-strategy"));

      if (gradient_strategy->isNumerical()) {
        gradient_strategy->setFunctionValue(
            val - std::real(observable->getIdentitySubTerm()->coefficient()));
      }

      gradient_strategy->initialize(options);
      auto grad_kernels = gradient_strategy->getGradientExecutions(
          kernel, current_iterate_parameters);

      auto tmp_grad = qalloc(qreg.size());
      qpu->execute(xacc::as_shared_ptr(tmp_grad.results()), grad_kernels);
      auto tmp_grad_children = tmp_grad.results()->getChildren();
      gradient_strategy->compute(dx, tmp_grad_children);
    }
    return val;
  }

public:
  const std::string name() const override { return "vqe"; }
  const std::string description() const override { return ""; }
};

} // namespace qcor

namespace {

/**
 */
class US_ABI_LOCAL VQEObjectiveActivator : public BundleActivator {

public:
  VQEObjectiveActivator() {}

  /**
   */
  void Start(BundleContext context) {
    auto xt = std::make_shared<qcor::VQEObjective>();
    context.RegisterService<qcor::ObjectiveFunction>(xt);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(VQEObjectiveActivator)
