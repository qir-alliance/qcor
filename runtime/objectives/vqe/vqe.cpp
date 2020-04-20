#include "qcor.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

#include <memory>
#include <set>

using namespace cppmicroservices;


namespace qcor {

class VQE : public ObjectiveFunction {
protected:
  double operator()() override {
    auto tmp_child = qalloc(qreg.size());
    auto val= qcor::__internal__::observe(kernel, observable, tmp_child);
    qreg.addChild(tmp_child);
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
