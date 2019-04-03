#ifndef COMPILER_QCORASTVISITOR_HPP_
#define COMPILER_QCORASTVISITOR_HPP_

#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Tooling/Tooling.h"

#include "QCORASTConsumer.hpp"

#include "XACC.hpp"

using namespace xacc;
using namespace clang;

namespace xacc {
class IRProvider;
}
namespace qcor {
namespace compiler {

class CallExprCloner : public RecursiveASTVisitor<CallExprCloner> {
public:
  LambdaExpr *cloned;
  bool VisitLambdaExpr(LambdaExpr *Node) {
      cloned = Node;
    return true;
  }
};

class ASTGeneratorAction : public clang::ASTFrontendAction {

protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef /* dummy */) override {
    return llvm::make_unique<compiler::QCORASTConsumer>(Compiler);
  }

  void ExecuteAction() override {
    CompilerInstance &CI = getCompilerInstance();
    CI.createSema(getTranslationUnitKind(), nullptr);
    ParseAST(CI.getSema());
  }

};

class LambdaVisitor : public RecursiveASTVisitor<LambdaVisitor> {

protected:
  class IsQuantumKernelVisitor
      : public RecursiveASTVisitor<IsQuantumKernelVisitor> {
  protected:
    ASTContext &context;
    bool _isQuantumKernel = false;
    std::vector<std::string> validInstructions;

  public:
    IsQuantumKernelVisitor(ASTContext &c);
    bool VisitDeclRefExpr(DeclRefExpr *expr);
    bool isQuantumKernel() { return _isQuantumKernel; }
  };

  class CppToXACCIRVisitor : public RecursiveASTVisitor<CppToXACCIRVisitor> {
  protected:
    ASTContext &context;

    std::shared_ptr<Function> function;
    std::shared_ptr<xacc::IRProvider> provider;
    std::vector<std::string> irGeneratorNames;

    std::string gateName = "";
    std::vector<int> bits;
    std::vector<InstructionParameter> parameters;

  public:
    CppToXACCIRVisitor(ASTContext &c);
    bool VisitCallExpr(CallExpr *expr);
    bool VisitDeclRefExpr(DeclRefExpr *expr);
    bool VisitIntegerLiteral(IntegerLiteral *literal);
    bool VisitFloatingLiteral(FloatingLiteral *literal);
    bool VisitParmVarDecl(ParmVarDecl *decl);
    std::shared_ptr<Function> getFunction();
  };

public:
  LambdaVisitor(CompilerInstance &c);

  bool VisitLambdaExpr(LambdaExpr *LE);

private:
  CompilerInstance &ci;
};
} // namespace compiler
} // namespace qcor
#endif
