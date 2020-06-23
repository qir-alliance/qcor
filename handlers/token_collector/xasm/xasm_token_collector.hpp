#ifndef QCOR_HANDLERS_XASMTOKENCOLLECTOR_HPP_
#define QCOR_HANDLERS_XASMTOKENCOLLECTOR_HPP_

#include "token_collector.hpp"

namespace qcor {
class XasmTokenCollector : public TokenCollector {
public:
  void collect(clang::Preprocessor &PP, clang::CachedTokens &Toks,
               std::vector<std::string> bufferNames,
               std::stringstream &ss) override;
  const std::string name() const override { return "xasm"; }
  const std::string description() const override { return ""; }
};

} // namespace qcor

#endif