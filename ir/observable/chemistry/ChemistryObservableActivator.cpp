#include "ChemistryObservable.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

#include <memory>
#include <set>

using namespace cppmicroservices;

namespace {

/**
 */
class US_ABI_LOCAL ChemistryObservableActivator : public BundleActivator {

public:
  ChemistryObservableActivator() {}

  /**
   */
  void Start(BundleContext context) {
    auto c = std::make_shared<qcor::observable::ChemistryObservable>();
    context.RegisterService<xacc::Observable>(c);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(ChemistryObservableActivator)
