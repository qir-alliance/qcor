#pragma once

#include "clang/Parse/Parser.h"

using namespace clang;

namespace qcor {


class Qasm3SyntaxHandler : public SyntaxHandler {
public:
  Qasm3SyntaxHandler() : SyntaxHandler("qasm3") {}
  void GetReplacement(Preprocessor &PP, Declarator &D, CachedTokens &Toks,
                      llvm::raw_string_ostream &OS) override;
  
 
  void AddToPredefines(llvm::raw_string_ostream &OS) override;
};
} // namespace qcor