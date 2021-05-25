#include "AlgorithmGradientStrategy.hpp"
#include "cppmicroservices/ServiceProperties.h"
#include "gradient_function.hpp"
#include "qcor.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_plugin.hpp"
#include "xacc_service.hpp"
using namespace cppmicroservices;

namespace qcor {
class KernelForwardDifferenceGradient : public KernelGradientService {
protected:
  std::shared_ptr<ObjectiveFunction> m_objFunc;
  double m_step = 1e-7;

public:
  const std::string name() const override { return "forward"; }
  const std::string description() const override { return ""; }
  void initialize(
      std::shared_ptr<ObjectiveFunction> obj_func,
      std::function<std::shared_ptr<CompositeInstruction>(std::vector<double>)>
          &kernel_eval,
      HeterogeneousMap &&options) override {
    if (options.keyExists<double>("step")) {
      m_step = options.get<double>("step");
    }

    auto observable = m_objFunc->get_observable();
    gradient_func = [&](const std::vector<double> &x,
                        double cost_val) -> std::vector<double> {
      std::vector<double> gradients(x.size(), 0.0);
      // TODO: port the implementation here as well.
      auto gradient_strategy =
          xacc::getService<xacc::AlgorithmGradientStrategy>("forward");
      if (gradient_strategy->isNumerical() &&
          observable->getIdentitySubTerm()) {
        gradient_strategy->setFunctionValue(
            cost_val -
            std::real(observable->getIdentitySubTerm()->coefficient()));
      }
      auto kernel = kernel_eval(x);
      gradient_strategy->initialize({{"observable", observable},
                                     {"step", m_step},
                                     {"kernel-evaluator", kernel_eval}});
      auto grad_kernels = gradient_strategy->getGradientExecutions(kernel, x);
      const size_t nb_qubits = std::max(
          static_cast<size_t>(observable->nBits()), kernel->nPhysicalBits());
      auto tmp_grad = qalloc(nb_qubits);
      xacc::internal_compiler::execute(tmp_grad.results(), grad_kernels);
      auto tmp_grad_children = tmp_grad.results()->getChildren();
      gradient_strategy->compute(gradients, tmp_grad_children);
      return gradients;
    };
  }
};
} // namespace qcor
REGISTER_PLUGIN(qcor::KernelForwardDifferenceGradient, qcor::KernelGradientService)
