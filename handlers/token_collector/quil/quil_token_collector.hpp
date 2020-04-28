#ifndef QCOR_HANDLERS_QUILTOKENCOLLECTOR_HPP_
#define QCOR_HANDLERS_QUILTOKENCOLLECTOR_HPP_

#include "token_collector.hpp"

namespace qcor {
class QuilTokenCollector : public TokenCollector {
public:
  void collect(clang::Preprocessor &PP, clang::CachedTokens &Toks,
               std::stringstream &ss) override;
  const std::string name() const override { return "quil"; }
  const std::string description() const override { return ""; }
};

} // namespace qcor

#endif