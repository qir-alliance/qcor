#pragma once

#include "clang/Parse/Parser.h"

using namespace clang;

namespace qcor {
extern std::string qpu_name;
extern int shots;

class QCORSyntaxHandler : public SyntaxHandler {
public:
  QCORSyntaxHandler() : SyntaxHandler("qcor") {}
  void GetReplacement(Preprocessor &PP, Declarator &D, CachedTokens &Toks,
                      llvm::raw_string_ostream &OS) override;
  
  // For use with qcor jit
  void GetReplacement(Preprocessor &PP, std::string &kernel_name,
                      std::vector<std::string> program_arg_types,
                      std::vector<std::string> program_parameters,
                      std::vector<std::string> bufferNames, CachedTokens &Toks,
                      llvm::raw_string_ostream &OS);

  void AddToPredefines(llvm::raw_string_ostream &OS) override;
};
} // namespace qcor