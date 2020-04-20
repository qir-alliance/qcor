#include "xasm_token_collector.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
#include "clang/Basic/TokenKinds.h"

#include <memory>
#include <set>
#include <iostream> 

using namespace cppmicroservices;

namespace {

/**
 */
class US_ABI_LOCAL XasmTokenCollectorActivator : public BundleActivator {

public:
  XasmTokenCollectorActivator() {}

  /**
   */
  void Start(BundleContext context) {
    auto xt = std::make_shared<qcor::XasmTokenCollector>();
    context.RegisterService<qcor::TokenCollector>(xt);
    // context.RegisterService<xacc::OptionsProvider>(acc);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(XasmTokenCollectorActivator)

namespace qcor {

void XasmTokenCollector::collect(clang::Preprocessor &PP,
                                 clang::CachedTokens &Toks,
                                 std::stringstream &ss) {
  bool inForLoop = false;
  for (auto &Tok : Toks) {
    if (PP.getSpelling(Tok) == "for") {
        inForLoop = true;
    }

    if (inForLoop && Tok.is(clang::tok::TokenKind::l_brace)) {
        inForLoop = false;
    }

    if (inForLoop) {
        ss << PP.getSpelling(Tok) << " ";
    } else {
        ss << PP.getSpelling(Tok);
    }
  }
}



} // namespace qcor