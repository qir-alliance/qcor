#include "/home/project/qcor/instructions/hwe/hwe.hpp"
#include "YPulse.hpp"
#include "AMeas.hpp"
#include "hwe.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

using namespace cppmicroservices;

class US_ABI_LOCAL QCORInstructionActivator : public BundleActivator {
public:
  QCORInstructionActivator() {}

  void Start(BundleContext context) {
    auto inst = std::make_shared<xacc::quantum::YPulse>();
    auto inst2 = std::make_shared<xacc::quantum::AMeas>();
    auto hwe = std::make_shared<qcor::instructions::HWE>();

    context.RegisterService<xacc::quantum::GateInstruction>(inst);
    context.RegisterService<xacc::quantum::GateInstruction>(inst2);
    context.RegisterService<xacc::IRGenerator>(hwe);

  }

  void Stop(BundleContext context) {}
};

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(QCORInstructionActivator)
