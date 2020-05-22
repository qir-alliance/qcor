#include "qcor.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

#include <memory>
#include <set>

using namespace cppmicroservices;


namespace qcor {

class QAOA : public ObjectiveFunction {
protected:
  double operator()() override {
    // TODO
    return 0.0;
  }
public:
  const std::string name() const override { return "qaoa"; }
  const std::string description() const override { return ""; }
};


} // namespace qcor

namespace {

/**
 */
class US_ABI_LOCAL QAOAObjectiveActivator : public BundleActivator {

public:
  QAOAObjectiveActivator() {}

  /**
   */
  void Start(BundleContext context) {
    auto xt = std::make_shared<qcor::QAOA>();
    context.RegisterService<qcor::ObjectiveFunction>(xt);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(QAOAObjectiveActivator)
