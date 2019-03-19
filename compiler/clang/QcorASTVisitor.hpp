#ifndef COMPILER_QCORASTVISITOR_HPP_
#define COMPILER_QCORASTVISITOR_HPP_

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "QuantumKernelHandler.hpp"

#include "clang/Parse/ParseAST.h"

#include "XACC.hpp"
#include "IRProvider.hpp"

using namespace clang;
using namespace xacc;

namespace qcor {
static const SourceLocation nopos;

class CppToXACCIRVisitor : public RecursiveASTVisitor<CppToXACCIRVisitor> {
protected:

   ASTContext& context;

   std::shared_ptr<Function> function;
   std::shared_ptr<IRProvider> provider;

   std::string gateName = "";
   std::vector<int> bits;
   std::vector<InstructionParameter> parameters;

public:
   CppToXACCIRVisitor(ASTContext& c): context(c) {
       provider = xacc::getService<IRProvider>("gate");
       function = provider->createFunction("tmp", {});
   }

   bool VisitCallExpr(CallExpr* expr) {
       if (gateName != "") {
           // create the gate instruction
           if (gateName == "CX") {gateName = "CNOT";}

           auto inst = provider->createInstruction(gateName, bits, parameters);
           function->addInstruction(inst);
       }
       gateName = "";
       bits.clear();
       parameters.clear();
       return true;
   }

   bool VisitDeclRefExpr(DeclRefExpr* expr) {

       if (expr->getType() == context.DependentTy) {
         gateName = expr->getNameInfo().getAsString();

        //  std::cout << "VISITING " << gateName << ", " << expr->getType().getAsString() << "\n";
       } else if (expr->getType() == context.DoubleTy) {
           InstructionParameter p(expr->getNameInfo().getAsString());
           parameters.push_back(p);
       }

       return true;
   }

   bool VisitIntegerLiteral(IntegerLiteral* literal) {
       bits.push_back(literal->getValue().getLimitedValue());
       return true;
   }

   bool VisitFloatingLiteral(FloatingLiteral* literal) {
       InstructionParameter p(literal->getValue().convertToDouble());
       parameters.push_back(p);
       return true;
   }

   std::shared_ptr<Function> getFunction() {
       // add the last one
        if (gateName != "") {
           // create the gate instruction
           if (gateName == "CX") {gateName = "CNOT";}

           auto inst = provider->createInstruction(gateName, bits, parameters);
           function->addInstruction(inst);
       }
       return function;
   }
};

class QcorASTVisitor : public RecursiveASTVisitor<QcorASTVisitor> {

public:

  QcorASTVisitor(CompilerInstance &c) : ci(c) {}

  bool VisitLambdaExpr(LambdaExpr *LE) {
    SourceManager &SM = ci.getSourceManager();
    LangOptions &lo = ci.getLangOpts();
    lo.CPlusPlus11 = 1;
    auto xaccKernelLambdaStr =
        Lexer::getSourceText(CharSourceRange(LE->getSourceRange(), true), SM,
                             lo)
            .str();

    std::cout << "Check it out, I got the Lambda as a source string :)\n";
    xacc::info(xaccKernelLambdaStr);

    CppToXACCIRVisitor visitor(ci.getASTContext());

    //  LE->getType().dump();

    // create empty statement, does nothing
    // Stmt *tmp = (Stmt *)new (ci.getASTContext()) NullStmt(nopos);
    // std::vector<Stmt *> stmts;
    // stmts.push_back(tmp);
    // replace LE's compound statement with that statement
    // LE->getBody()->setLastStmt(ReturnStmt::CreateEmpty(
    //     ci.getASTContext(),
    //     false));

    Stmt* q_kernel_body = LE->getBody();
    q_kernel_body->dump();

    std::cout << "TRAVERSING STMT:\n";
    visitor.TraverseStmt(LE->getBody());
    auto function = visitor.getFunction();
    std::cout << "XACC IR Qasm:\n" << function->toString() << "\n";


    return true;
  }

private:
  CompilerInstance &ci;
};

} // namespace qcor
#endif

    // const auto& parents = ci.getASTContext().getParents(old_stmt);
    // auto it = parents.begin();
    // if (it == parents.end()) std::cout << "NO PARENTS:\n";
    // else {
    //     std::cout << "this has " << parents.size() << " parents\n";

    //     const Stmt* parent = parents[0].get<Stmt>();
    //     parent->dump();

    //     Stmt& new_stmt = *ReturnStmt::CreateEmpty(
    //        ci.getASTContext(),
    //        false);

    //     // Stmt::const_child_iterator it = std::find(parent->child_begin(), parent->child_end(), old_stmt);
    //     // *it = new_stmt;
    //     // std::replace(parent->child_begin(), parent->child_end(), old_stmt, new_stmt);
    // }