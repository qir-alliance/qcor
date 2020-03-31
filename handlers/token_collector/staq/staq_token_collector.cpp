#include "staq_token_collector.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

#include <iostream>
#include <memory>
#include <set>

#include "clang/Parse/Parser.h"

using namespace cppmicroservices;
using namespace clang;

namespace {

/**
 */
class US_ABI_LOCAL StaqTokenCollectorActivator : public BundleActivator {

public:
  StaqTokenCollectorActivator() {}

  /**
   */
  void Start(BundleContext context) {
    auto st = std::make_shared<qcor::StaqTokenCollector>();
    context.RegisterService<qcor::TokenCollector>(st);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(StaqTokenCollectorActivator)

namespace qcor {

static const std::map<std::string, std::string> gates{
    // "u3", "u2",   "u1", "ccx", cu1, cu3
    {"cx", "CX"},
    {"id", "I"},
    {"x", "X"},
    {"y", "Y"},
    {"z", "Z"},
    {"h", "H"},
    {"s", "S"},
    {"sdg", "Sdg"},
    {"t", "T"},
    {"tdg", "Tdg"},
    {"rx", "Rx"},
    {"ry", "Ry"},
    {"rz", "Rz"},
    {"cz", "CZ"},
    {"cy", "CY"},
    {"swap", "Swap"},
    {"ch", "CH"},
    {"crz", "CRZ"},
    {"measure", "Measure"}};

void StaqTokenCollector::collect(clang::Preprocessor &PP,
                                 clang::CachedTokens &Toks,
                                 std::stringstream &ss) {
  bool getOracleName = false;
  std::string oracleName;
  for (auto &Tok : Toks) {
    // std::cout << PP.getSpelling(Tok) << "\n";
    //   ss << PP.getSpelling(Tok);
    if (getOracleName) {
      ss << " " << PP.getSpelling(Tok) << " ";
      oracleName = PP.getSpelling(Tok);
      getOracleName = false;
    } else if (PP.getSpelling(Tok) == "oracle") {
      ss << PP.getSpelling(Tok) << " ";
      getOracleName = true;
    } else if (Tok.is(tok::TokenKind::semi)) {
      ss << ";\n";
    } else if (Tok.is(tok::TokenKind::l_brace)) {
      // add space before lbrach
      ss << " " << PP.getSpelling(Tok) << " ";
    } else if (Tok.is(tok::TokenKind::r_brace)) {
      ss << " " << PP.getSpelling(Tok) << "\n";
    } else if (PP.getSpelling(Tok) == "creg" || PP.getSpelling(Tok) == "qreg") {
      ss << PP.getSpelling(Tok) << " ";
    } else if (gates.count(PP.getSpelling(Tok)) ||
               PP.getSpelling(Tok) == oracleName) {
      ss << PP.getSpelling(Tok) << " ";
    } else {
      ss << PP.getSpelling(Tok);
    }
  }
}

} // namespace qcor