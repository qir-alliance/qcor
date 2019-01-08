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

class GetEmptyLambdaVisitor : public RecursiveASTVisitor<GetEmptyLambdaVisitor> {
public:
  GetEmptyLambdaVisitor()  {}

  bool TraverseLambdaBody(LambdaExpr *LE) {
      le = LE;
      return true;
  }

  LambdaExpr* getLE() {return le;}
  private:
  LambdaExpr* le;
};

class GetEmptyLambdaConsumer : public ASTConsumer {
public:
  GetEmptyLambdaConsumer(CompilerInstance& c) : ci(c) {}

  // Override the method that gets called for each parsed top-level
  // declaration.
  bool HandleTopLevelDecl(DeclGroupRef DR) override {
    Visitor.TraverseDecl(ci.getASTContext().getTranslationUnitDecl());
    return true;
  }

  LambdaExpr* getLE() {return Visitor.getLE();}
  
private:
  CompilerInstance &ci;
  GetEmptyLambdaVisitor Visitor;
};

class QcorASTVisitor : public RecursiveASTVisitor<QcorASTVisitor> {
public:
  QcorASTVisitor(CompilerInstance &c) : ci(c) {}

  bool TraverseLambdaBody(LambdaExpr *LE) {
    SourceManager &SM = ci.getSourceManager();
    LangOptions &lo = ci.getLangOpts();
    lo.CPlusPlus11 = 1;
    auto xaccKernelLambdaStr =
        Lexer::getSourceText(CharSourceRange(LE->getSourceRange(), true), SM,
                             lo).str();

    std::cout << "Check it out, I got the Lambda as a source string :)\n";
    std::cout << xaccKernelLambdaStr << "\n";

     LE->getType().dump();
     
//   CompilerInstance tmpCi;
//   ci.getLangOpts() = lo;
//   DiagnosticOptions diagnosticOptions;
//   ci.createDiagnostics();

//   std::shared_ptr<clang::TargetOptions> pto =
//       std::make_shared<clang::TargetOptions>();
//   pto->Triple = llvm::sys::getDefaultTargetTriple();

//   TargetInfo *pti = TargetInfo::CreateTargetInfo(ci.getDiagnostics(), pto);

//   ci.setTarget(pti);
//   ci.createFileManager();
//   ci.createSourceManager(ci.getFileManager());
//   ci.createPreprocessor(clang::TU_Complete);
//   ci.createASTContext();
//   SourceManager &SourceMgr = ci.getSourceManager();

//   ci.setASTConsumer(llvm::make_unique<GetEmptyLambdaConsumer>(ci));
  
//   ci.createSema(TU_Complete, nullptr);
//   auto &sema = ci.getSema();
//   sema.Initialize();

//   std::ofstream of(".tmp_lambda_xacc.cpp");
//   of << "[&]() {\n}";
//   of.close();
  
//   const FileEntry *pFile = ci.getFileManager().getFile(".tmp_lambda_xacc.cpp");
//   ci.getSourceManager().setMainFileID(ci.getSourceManager().createFileID(
//       pFile, clang::SourceLocation(), clang::SrcMgr::C_User));
//   ci.getDiagnosticClient().BeginSourceFile(ci.getLangOpts(),
//                                            &ci.getPreprocessor());
//   clang::ParseAST(sema, true, false);


    return true;
  }

private:
  CompilerInstance &ci;
};

} // namespace qcor
#endif