#ifndef COMPILER_QCORASTVISITOR_HPP_
#define COMPILER_QCORASTVISITOR_HPP_

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"

#include "QuantumKernelHandler.hpp"

#include "clang/Parse/ParseAST.h"

using namespace clang;

namespace qcor {

class QcorASTVisitor : public RecursiveASTVisitor<QcorASTVisitor> {
public:
  QcorASTVisitor(CompilerInstance &c) : ci(c) {}

  bool VisitLambdaExpr(LambdaExpr *LE) {
    SourceManager &SM = ci.getSourceManager();
    LangOptions &lo = ci.getLangOpts();
    lo.CPlusPlus11 = 1;
    auto xaccKernelLambdaStr =
        Lexer::getSourceText(CharSourceRange(LE->getSourceRange(), true), SM,
                             lo).str();

    std::cout << "Check it out, I got the Lambda as a source string :)\n";
    xacc::info(xaccKernelLambdaStr);

    //  LE->getType().dump();

    return true;
  }

private:
  CompilerInstance &ci;
};

} // namespace qcor
#endif