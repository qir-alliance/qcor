#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
#include "qcor.hpp"
using namespace cppmicroservices;

#include <iomanip>
#include <memory>
#include <set>

#include "AlgorithmGradientStrategy.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_plugin.hpp"
#include "xacc_service.hpp"

namespace {

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  os << "[";
  for (int i = 0; i < v.size(); ++i) {
    os << v[i];
    if (i != v.size() - 1) os << ",";
  }
  os << "]";
  return os;
}
}  // namespace
namespace qcor {

class VQEObjective : public ObjectiveFunction {
 public:
  std::shared_ptr<xacc::Algorithm> vqe;
  double operator()(xacc::internal_compiler::qreg &qreg,
                    std::vector<double> &dx) override {
    if (!vqe) {
      vqe = xacc::getAlgorithm("vqe");
    }

    gradients_computed = false;
    auto xacc_kernel = kernel->as_xacc();
    auto xacc_observable =
        std::dynamic_pointer_cast<xacc::Observable>(observable.get_as_opaque());
    auto qpu = xacc::internal_compiler::get_qpu();
    auto success = vqe->initialize({{"ansatz", xacc_kernel},
                                    {"accelerator", qpu},
                                    {"observable", xacc_observable}});

    if (!success) {
      xacc::error(
          "QCOR VQE Error - could not initialize internal xacc vqe algorithm.");
    }

    auto tmp_child = std::make_shared<xacc::AcceleratorBuffer>("temp_vqe_child",
                                                               qreg.size());
    auto val = vqe->execute(tmp_child, {})[0];
    double std_dev = 0.0;
    if (options.keyExists<int>("vqe-gather-statistics")) {
      std::vector<double> all_energies;
      all_energies.push_back(val);
      auto n = options.get<int>("vqe-gather-statistics");
      for (int i = 1; i < n; i++) {
        auto tmp_child = qalloc(qreg.size());
        auto local_val =
            vqe->execute(xacc::as_shared_ptr(tmp_child.results()), {})[0];
        all_energies.push_back(local_val);
      }
      auto sum = std::accumulate(all_energies.begin(), all_energies.end(), 0.);
      val = sum / all_energies.size();
      double sq_sum = std::inner_product(
          all_energies.begin(), all_energies.end(), all_energies.begin(), 0.0);
      std_dev = std::sqrt(sq_sum / all_energies.size() - val * val);
    }

    if (options.keyExists<bool>("verbose") && options.get<bool>("verbose")) {
      std::cout << "<H>(" << this->current_iterate_parameters
                << ") = " << std::setprecision(12) << val;
      if (std::fabs(std_dev) > 1e-12) {
        std::cout << " +- " << std_dev << "\n";
      } else {
        std::cout << std::endl;
      }
    }

    // want to store parameters, have to do it here
    for (auto &child : tmp_child->getChildren()) {
      child->addExtraInfo("parameters", current_iterate_parameters);
      auto tmp = current_iterate_parameters;
      tmp.push_back(val);
      child->addExtraInfo("qcor-params-energy", tmp);
      if (std::fabs(std_dev) > 1e-12) {
        child->addExtraInfo("qcor-energy-stddev", std_dev);
      }
      child->addExtraInfo("iteration", current_iteration);
      qreg.results()->appendChild(child->name(), child);
    }
    current_iteration++;
    // qreg.addChild(tmp_child);

    if (!dx.empty() && options.stringExists("gradient-strategy")) {
      // Compute the gradient
      auto gradient_strategy =
          xacc::getService<xacc::AlgorithmGradientStrategy>(
              options.getString("gradient-strategy"));

      if (gradient_strategy->isNumerical() &&
          xacc_observable->getIdentitySubTerm()) {
        gradient_strategy->setFunctionValue(
            val -
            std::real(xacc_observable->getIdentitySubTerm()->coefficient()));
      }

      if (!options.key_exists_any_type("kernel-evaluator")) {
        error("cannot compute gradients without kernel evaluator.");
      }

      // First translate to an xacc kernel evaluator
      auto k_eval =
          options.get<std::function<std::shared_ptr<CompositeInstruction>(
              std::vector<double>)>>("kernel-evaluator");

      std::function<std::shared_ptr<xacc::CompositeInstruction>(
          std::vector<double>)>
          xacc_k_eval =
              [&](std::vector<double> x) { return k_eval(x)->as_xacc(); };

      auto step = options.get_or_default("step", 1e-3);
      gradient_strategy->initialize({{"kernel-evaluator", xacc_k_eval},
                                     {"observable", xacc_observable},
                                     {"step", step}});

      auto grad_kernels = gradient_strategy->getGradientExecutions(
          xacc_kernel, current_iterate_parameters);

      auto tmp_grad = qalloc(qreg.size());
      qpu->execute(xacc::as_shared_ptr(tmp_grad.results()), grad_kernels);
      auto tmp_grad_children = tmp_grad.results()->getChildren();
      gradient_strategy->compute(dx, tmp_grad_children);
      gradients_computed = true;
    }
    return val;
  }

 public:
  int current_iteration = 0;
  const std::string name() const override { return "vqe"; }
  const std::string description() const override { return ""; }
};

}  // namespace qcor

REGISTER_PLUGIN(qcor::VQEObjective, qcor::ObjectiveFunction)
