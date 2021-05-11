#ifndef QCOR_HANDLERS_UNITARYTOKENCOLLECTOR_HPP_
#define QCOR_HANDLERS_UNITARYTOKENCOLLECTOR_HPP_

#include "token_collector.hpp"

namespace qcor {
class UnitaryTokenCollector : public TokenCollector {
public:
  void collect(clang::Preprocessor &PP, clang::CachedTokens &Toks,
               std::vector<std::string> bufferNames,
               std::stringstream &ss, const std::string &kernel_name) override;
  const std::string name() const override { return "unitary"; }
  const std::string description() const override { return ""; }
};

} // namespace qcor

#endif