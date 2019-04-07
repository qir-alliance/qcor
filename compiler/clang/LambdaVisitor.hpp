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

#include "QCORASTConsumer.hpp"

#include "IRGenerator.hpp"
#include "XACC.hpp"

using namespace xacc;
using namespace clang;

namespace xacc {
class IRProvider;
}
namespace qcor {
namespace compiler {

class LambdaVisitor : public RecursiveASTVisitor<LambdaVisitor> {

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
    bool VisitLambdaExpr(LambdaExpr* expr);
    bool isQuantumKernel() { return _isQuantumKernel; }
  };

  class CppToXACCIRVisitor : public RecursiveASTVisitor<CppToXACCIRVisitor> {
  protected:
    std::shared_ptr<Function> function;
    std::shared_ptr<xacc::IRProvider> provider;
    std::vector<std::string> irGeneratorNames;

  public:
    CppToXACCIRVisitor();
    bool VisitCallExpr(CallExpr *expr);
    std::shared_ptr<Function> getFunction();
  };

  class CallExprToGateInstructionVisitor
      : public RecursiveASTVisitor<CallExprToGateInstructionVisitor> {
  protected:
    std::vector<int> bits;
    std::vector<InstructionParameter> parameters;
    std::string name;
    std::shared_ptr<IRProvider> provider;

  public:
    CallExprToGateInstructionVisitor(const std::string n,
                                     std::shared_ptr<IRProvider> p)
        : name(n), provider(p) {}
    std::shared_ptr<Instruction> getInstruction();
    bool VisitIntegerLiteral(IntegerLiteral *il);
    bool VisitFloatingLiteral(FloatingLiteral *fl);
    bool VisitDeclRefExpr(DeclRefExpr *expr);
  };

  class CallExprToIRGenerator
      : public RecursiveASTVisitor<CallExprToIRGenerator> {
  protected:
    std::shared_ptr<IRProvider> provider;
    std::string name;
    std::map<std::string, InstructionParameter> options;
    bool haveSeenFirstDeclRef = false;
    bool haveSeenFirstInit = false;

  public:
    CallExprToIRGenerator(const std::string n, std::shared_ptr<IRProvider> p)
        : name(n), provider(p) {}
    // bool VisitDeclRefExpr(DeclRefExpr* expr);
    bool VisitInitListExpr(InitListExpr *expr);
    std::shared_ptr<IRGenerator> getIRGenerator();
  };
  class ScanInitListExpr : public RecursiveASTVisitor<ScanInitListExpr> {
  protected:
    bool isFirstStringLiteral = true;
    // bool isSubInit;
  public:
    // std::vector<int> intsFound;
    // std::vector<double> realsFound;
    std::string key;
    InstructionParameter value;
    // ScanInitListExpr(bool isSubInitList = false) :isSubInit(isSubInitList) {}
    bool VisitDeclRefExpr(DeclRefExpr *expr);
    bool VisitStringLiteral(StringLiteral *literal);
    bool VisitFloatingLiteral(FloatingLiteral *literal);
    bool VisitIntegerLiteral(IntegerLiteral *literal);
    // bool VisitInitListExpr(InitListExpr *initList);
  };

public:
  LambdaVisitor(CompilerInstance &c, Rewriter &rw);

  bool VisitLambdaExpr(LambdaExpr *LE);

private:
  CompilerInstance &ci;
  Rewriter &rewriter;
};
} // namespace compiler
} // namespace qcor
#endif
