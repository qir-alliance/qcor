#ifndef QCOR_HANDLERS_TOKENCOLLECTOR_HPP_
#define QCOR_HANDLERS_TOKENCOLLECTOR_HPP_

#include "Identifiable.hpp"
#include "clang/Parse/Parser.h"
#include <sstream>

namespace qcor {
class TokenCollector : public xacc::Identifiable {
public:
  virtual void collect(clang::Preprocessor &PP, clang::CachedTokens &Toks,
                       std::stringstream &ss) = 0;
};

} // namespace qcor

#endif