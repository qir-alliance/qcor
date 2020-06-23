#ifndef QCOR_HANDLERS_STAQTOKENCOLLECTOR_HPP_
#define QCOR_HANDLERS_STAQTOKENCOLLECTOR_HPP_

#include "token_collector.hpp"

namespace qcor {
class StaqTokenCollector : public TokenCollector {
public:
  void collect(clang::Preprocessor &PP, clang::CachedTokens &Toks,
               std::vector<std::string> bufferNames,
               std::stringstream &ss) override;
  const std::string name() const override { return "staq"; }
  const std::string description() const override { return ""; }
};

} // namespace qcor

#endif