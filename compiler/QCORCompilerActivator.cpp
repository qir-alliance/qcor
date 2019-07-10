#include "QCORCompiler.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "clang/FuzzyParsingExternalSemaSource.hpp"
#include "clang/QCORExternalSemaSource.hpp"

#include <memory>
#include <set>

using namespace cppmicroservices;

namespace {

/**
 */
class US_ABI_LOCAL QCORActivator : public BundleActivator {

public:
  QCORActivator() {}

  /**
   */
  void Start(BundleContext context) {

    auto s = std::make_shared<qcor::QCORCompiler>();
    context.RegisterService<xacc::Compiler>(s);

    auto es = std::make_shared<qcor::compiler::FuzzyParsingExternalSemaSource>();
    context.RegisterService<qcor::compiler::QCORExternalSemaSource>(es);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(QCORActivator)
