#include "hwe.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

using namespace cppmicroservices;

class US_ABI_LOCAL QCORDigitalActivator : public BundleActivator {
public:
  QCORDigitalActivator() {}

  void Start(BundleContext context) {
    auto hwe = std::make_shared<qcor::instructions::HWE>();
    context.RegisterService<xacc::IRGenerator>(hwe);
  }

  void Stop(BundleContext context) {}
};

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(QCORDigitalActivator)
