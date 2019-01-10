#include "YPulse.hpp"
#include "AMeas.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

using namespace cppmicroservices;

class US_ABI_LOCAL QcorInstructionActivator : public BundleActivator {
public:
  QcorInstructionActivator() {}

  void Start(BundleContext context) {
    auto inst = std::make_shared<xacc::quantum::YPulse>();
    auto inst2 = std::make_shared<xacc::quantum::AMeas>();

    context.RegisterService<xacc::quantum::GateInstruction>(inst);
    context.RegisterService<xacc::quantum::GateInstruction>(inst2);

  }

  void Stop(BundleContext context) {}
};

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(QcorInstructionActivator)
