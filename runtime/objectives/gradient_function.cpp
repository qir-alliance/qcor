#define US_BUNDLE_NAME
#include "gradient_function.hpp"
#include "AlgorithmGradientStrategy.hpp"
#include "cppmicroservices/ServiceProperties.h"
#include "qcor.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_plugin.hpp"
#include "xacc_service.hpp"
using namespace cppmicroservices;

namespace qcor {
KernelForwardDifferenceGradient::KernelForwardDifferenceGradient(
    std::function<std::shared_ptr<xacc::CompositeInstruction>(
        std::vector<double>)> &kernel_evaluator,
    std::shared_ptr<xacc::Observable> observable, double step_size)
    : m_kernelEval(kernel_evaluator), m_step(step_size), m_obs(observable) {
  gradient_func = [&](const std::vector<double> &x,
                      double cost_val) -> std::vector<double> {
    std::vector<double> gradients(x.size(), 0.0);
    // TODO: port the implementation here as well.
    auto gradient_strategy =
        xacc::getService<xacc::AlgorithmGradientStrategy>("forward");

    if (gradient_strategy->isNumerical() && m_obs->getIdentitySubTerm()) {
      gradient_strategy->setFunctionValue(
          cost_val - std::real(m_obs->getIdentitySubTerm()->coefficient()));
    }
    auto kernel = m_kernelEval(x);
    gradient_strategy->initialize({{"observable", m_obs}, {"step", m_step}});
    auto grad_kernels = gradient_strategy->getGradientExecutions(kernel, x);
    const size_t nb_qubits =
        std::max(static_cast<size_t>(m_obs->nBits()), kernel->nPhysicalBits());
    auto tmp_grad = qalloc(nb_qubits);
    xacc::internal_compiler::execute(tmp_grad.results(), grad_kernels);
    auto tmp_grad_children = tmp_grad.results()->getChildren();
    gradient_strategy->compute(gradients, tmp_grad_children);
    return gradients;
  };
}
} // namespace qcor
