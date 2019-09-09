#ifndef COMPILER_FUZZYPARSINGEXTERNALSEMASOURCE_HPP_
#define COMPILER_FUZZYPARSINGEXTERNALSEMASOURCE_HPP_

#include "clang/AST/ASTContext.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/Lookup.h"

using namespace clang;

namespace qcor {
namespace compiler {
class FuzzyParsingExternalSemaSource : public ExternalSemaSource {
private:
  std::vector<std::string> validInstructions;
  CompilerInstance &ci;
  ParmVarDecl *qbit;
  ParmVarDecl *hMapRValue;
  ParmVarDecl *stdVector;
  std::unique_ptr<ASTUnit> hast;
  std::vector<std::string> compositeInstructions;
//   std::vector<bool> compositeRequiresStdVector;

  // Keep a vector of ASTs for each FunctionDecl
  // representation of our quantum instructions.
  // This ExternalSemaSource should exist throughout
  // the tooling lifetime, so we should be good with
  // regards to these nodes being deleted
  std::vector<std::unique_ptr<ASTUnit>> quantumInstructionASTs;
  std::map<std::string, std::string> quantumInstruction2src;

public:
  FuzzyParsingExternalSemaSource(CompilerInstance &c) : ci(c) {}
  void initialize();
  //   void setASTContext(ASTContext *context) { m_Context = context; }
  //   void setFileManager(FileManager *m) { manager = m; }

  bool LookupUnqualified(clang::LookupResult &R, clang::Scope *S) override;
};
} // namespace compiler
} // namespace qcor
#endif