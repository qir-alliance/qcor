#ifndef COMPILER_FUZZYPARSINGEXTERNALSEMASOURCE_HPP_
#define COMPILER_FUZZYPARSINGEXTERNALSEMASOURCE_HPP_

#include "clang/AST/ASTContext.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/Lookup.h"

using namespace clang;

namespace qcor {
namespace compiler {
class FuzzyParsingExternalSemaSource : public ExternalSemaSource {
private:
  std::vector<std::string> validInstructions;
  ASTContext *m_Context;

  // Keep a vector of ASTs for each FunctionDecl
  // representation of our quantum instructions.
  // This ExternalSemaSource should exist throughout
  // the tooling lifetime, so we should be good with
  // regards to these nodes being deleted
  std::map<std::string, std::unique_ptr<ASTUnit>> quantumInstructionASTs;

public:
  FuzzyParsingExternalSemaSource() = default;
  void initialize();
  void setASTContext(ASTContext *context) { m_Context = context; }

  bool LookupUnqualified(clang::LookupResult &R, clang::Scope *S) override;
};
} // namespace compiler
} // namespace qcor
#endif