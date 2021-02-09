#include "qcor.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
using namespace cppmicroservices;

#include <memory>
#include <set>
#include <iomanip> 

#include "AlgorithmGradientStrategy.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_service.hpp"
#include "xacc_plugin.hpp"

namespace {

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  os << "[";
  for (int i = 0; i < v.size(); ++i) {
    os << v[i];
    if (i != v.size() - 1)
      os << ",";
  }
  os << "]";
  return os;
}
} // namespace
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

    auto tmp_child = std::make_shared<xacc::AcceleratorBuffer>("temp_vqe_child", qreg.size());
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

    std::cout << "<H>(" << this->current_iterate_parameters << ") = " << std::setprecision(12) << val;
    if (std::fabs(std_dev) > 1e-12) {
      std::cout << " +- " << std_dev << "\n";
    } else {
      std::cout << std::endl;
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
  int current_iteration = 0;
  const std::string name() const override { return "vqe"; }
  const std::string description() const override { return ""; }
};

} // namespace qcor

REGISTER_PLUGIN(qcor::VQEObjective, qcor::ObjectiveFunction)
