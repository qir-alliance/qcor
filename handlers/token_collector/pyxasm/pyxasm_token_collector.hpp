#pragma once

#include "token_collector.hpp"

namespace qcor {
class PyXasmTokenCollector : public TokenCollector {
public:
  void collect(clang::Preprocessor &PP, clang::CachedTokens &Toks,
               std::vector<std::string> bufferNames,
               std::stringstream &ss, const std::string &kernel_name) override;
  const std::string name() const override { return "pyxasm"; }
  const std::string description() const override { return ""; }
};

} // namespace qcor

