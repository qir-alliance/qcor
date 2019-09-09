#ifndef COMPILER_QCORASTVISITOR_HPP_
#define COMPILER_QCORASTVISITOR_HPP_

#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Tooling.h"

#include "xacc.hpp"

using namespace xacc;
using namespace clang;

namespace xacc {
class IRProvider;
}
namespace qcor {
namespace compiler {
class make_pair_visitor : public visitor_base<int, double, std::string> {
protected:
  std::stringstream &s;
  std::vector<std::string> captures;

public:
  make_pair_visitor(std::stringstream &ss, std::vector<std::string> &c)
      : s(ss), captures(c) {}
  template <typename T> void operator()(const std::string &key, const T &t) {
    if (std::find(captures.begin(), captures.end(), key) !=
        std::end(captures)) {
      s << "std::make_pair(\"" << key << "\"," << t << "),";
    } else {
      s << "std::make_pair(\"" << key << "\",\"" << t << "\"),";
    }
  }
};

class QCORASTVisitor : public RecursiveASTVisitor<QCORASTVisitor> {

protected:
  class IsQuantumKernelVisitor
      : public RecursiveASTVisitor<IsQuantumKernelVisitor> {
  protected:
    ASTContext &context;
    bool _isQuantumKernel = false;
    std::vector<std::string> validInstructions;
    bool foundSubLambda = false;

  public:
    IsQuantumKernelVisitor(ASTContext &c);
    bool VisitDeclRefExpr(DeclRefExpr *expr);
    bool VisitLambdaExpr(LambdaExpr *expr);
    bool isQuantumKernel() { return _isQuantumKernel; }
    std::string irType = "gate";
  };

public:
  QCORASTVisitor(CompilerInstance &c, Rewriter &rw);

  bool VisitLambdaExpr(LambdaExpr *LE);
  bool VisitFunctionDecl(FunctionDecl *decl);

private:
  CompilerInstance &ci;
  Rewriter &rewriter;
};
} // namespace compiler
} // namespace qcor
#endif
