#include "qcor.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

#include <memory>
#include <set>

#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"

using namespace cppmicroservices;

namespace qcor {

class VQE : public ObjectiveFunction {
protected:
  std::shared_ptr<xacc::Algorithm> vqe;
  double operator()() override {
    if (!vqe) {
      vqe = xacc::getAlgorithm("vqe");
    }
    vqe->initialize(
        {std::make_pair("ansatz", kernel),
         std::make_pair("accelerator", xacc::internal_compiler::get_qpu()),
         std::make_pair("observable", observable)});

    auto tmp_child = qalloc(qreg.size());
    auto val = vqe->execute(xacc::as_shared_ptr(tmp_child.results()), {})[0];
    qreg.addChild(tmp_child);
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
    auto xt = std::make_shared<qcor::VQE>();
    context.RegisterService<qcor::ObjectiveFunction>(xt);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(VQEObjectiveActivator)
