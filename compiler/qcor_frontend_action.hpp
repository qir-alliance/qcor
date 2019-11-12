#ifndef COMPILER_QCORFRONTENDACTION_HPP_
#define COMPILER_QCORFRONTENDACTION_HPP_

#include "clang/Frontend/FrontendAction.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Rewrite/Frontend/Rewriters.h"

#include "clang/Tooling/Tooling.h"
#include <fstream>
#include <string>

#include "fuzzy_parsing.hpp"

#include "qcor_ast_consumer.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"

using namespace clang;

namespace qcor {
namespace compiler {
class QCORFrontendAction : public clang::ASTFrontendAction {
public:
  QCORFrontendAction(Rewriter &rw, const std::string file, std::vector<std::string> args)
      : rewriter(rw), fileName(file), extraArgs(args) {}

protected:
  Rewriter &rewriter;
  std::string fileName;
  std::vector<std::string> extraArgs;
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef /* dummy */) override {
    return std::make_unique<qcor::compiler::QCORASTConsumer>(Compiler,
                                                             rewriter);
  }

  void ExecuteAction() override;
};
} // namespace compiler
} // namespace qcor
#endif
