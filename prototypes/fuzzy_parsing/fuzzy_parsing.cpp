//------------------------------------------------------------------------------
// Tooling sample. Demonstrates:
//
// * How to write a simple source tool using libTooling.
// * How to use RecursiveASTVisitor to find interesting AST nodes.
// * How to use the Rewriter API to rewrite the source code.
//
// Eli Bendersky (eliben@gmail.com)
// This code is in the public domain
//------------------------------------------------------------------------------
#include <sstream>
#include <string>
#include <iostream>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"

#include <clang/Frontend/FrontendActions.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/CommandLine.h>
#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <llvm/Support/raw_ostream.h>
#include <clang/Sema/ExternalSemaSource.h>
#include <clang/Sema/Sema.h>
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Parse/Parser.h"
#include "clang/Parse/ParseAST.h"
#include <clang/Sema/Lookup.h>

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

static llvm::cl::OptionCategory ToolingSampleCategory("Tooling Sample");

// By implementing RecursiveASTVisitor, we can specify which AST nodes
// we're interested in by overriding relevant methods.
class MyASTVisitor : public RecursiveASTVisitor<MyASTVisitor> {
public:
  MyASTVisitor(Rewriter &R, CompilerInstance& c) : TheRewriter(R), TheCompInst(c) {}

  bool VisitStmt(Stmt *s) {
    // Only care about If statements.
    if (isa<IfStmt>(s)) {
      IfStmt *IfStatement = cast<IfStmt>(s);
      Stmt *Then = IfStatement->getThen();

      TheRewriter.InsertText(Then->getBeginLoc(), "// the 'if' part\n", true,
                             true);

      Stmt *Else = IfStatement->getElse();
      if (Else)
        TheRewriter.InsertText(Else->getBeginLoc(), "// the 'else' part\n",
                               true, true);
    }

    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *f) {
      
    // Only function definitions (with bodies), not declarations.
    if (f->hasBody()) {
      Stmt *FuncBody = f->getBody();

      // Type name as string
      QualType QT = f->getReturnType();
      std::string TypeStr = QT.getAsString();

      // Function name
      DeclarationName DeclName = f->getNameInfo().getName();
      std::string FuncName = DeclName.getAsString();

      // Add comment before
      std::stringstream SSBefore;
      SSBefore << "// Begin function " << FuncName << " returning " << TypeStr
               << "\n";
      SourceLocation ST = f->getSourceRange().getBegin();
      TheRewriter.InsertText(ST, SSBefore.str(), true, true);

      // And after
      std::stringstream SSAfter;
      SSAfter << "\n// End function " << FuncName;
      ST = FuncBody->getEndLoc().getLocWithOffset(1);
      TheRewriter.InsertText(ST, SSAfter.str(), true, true);
    }

    return true;
  }
bool TraverseLambdaBody(LambdaExpr *LE) {
    SourceManager &SM = TheRewriter.getSourceMgr();
    LangOptions &lo = TheCompInst.getLangOpts();
    lo.CPlusPlus11 = 1;
  
    //   LE->getLambdaClass()->dump();
      std::cout << "Check it out, I got the Lambda as a source string :)\n";

      std::cout << Lexer::getSourceText(CharSourceRange(LE->getSourceRange(), true), SM, lo).str() << "\n";
        return true;
    }
private:
  Rewriter &TheRewriter;
  CompilerInstance& TheCompInst;
};


class DynamicIDHandler : public clang::ExternalSemaSource {
public:
  DynamicIDHandler(Sema * Sema): m_Context(Sema->getASTContext()) {} 
  /// \brief Provides last resort lookup for failed unqualified lookups
  ///
  /// If there is failed lookup, tell sema to create an artificial declaration
  /// which is of dependent type. So the lookup result is marked as dependent
  /// and the diagnostics are suppressed. After that is's an interpreter's
  /// responsibility to fix all these fake declarations and lookups.
  /// It is done by the DynamicExprTransformer.
  ///
  /// @param[out] R The recovered symbol.
  /// @param[in] S The scope in which the lookup failed.
  virtual bool LookupUnqualified(clang::LookupResult &R, clang::Scope *S) {
     DeclarationName Name = R.getLookupName();
     if (Name.getAsString() == "X") {
     std::cout << "Looking Up Unqualified " << Name.getAsString() << "\n";
    IdentifierInfo *II = Name.getAsIdentifierInfo();
    SourceLocation Loc = R.getNameLoc();
    VarDecl *Result =
        VarDecl::Create(m_Context, R.getSema().getFunctionLevelDeclContext(),
                        Loc, Loc, II, m_Context.DependentTy,
                        /*TypeSourceInfo*/ 0, SC_None);
    if (Result) {
      R.addDecl(Result);
      // Say that we can handle the situation. Clang should try to recover
      return true;
    } else{
      return false;
    }
  }
    return false;
  }

private:
  clang::ASTContext &m_Context;

};

// Implementation of the ASTConsumer interface for reading an AST produced
// by the Clang parser.
class MyASTConsumer : public ASTConsumer {
public:
  MyASTConsumer(Rewriter &R, CompilerInstance& c) : Visitor(R,c), CI(c) {}

  // Override the method that gets called for each parsed top-level
  // declaration.
  bool HandleTopLevelDecl(DeclGroupRef DR) override {
    for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b) {
      // Traverse the declaration using our AST visitor.
      Visitor.TraverseDecl(*b);
      (*b)->dump();
    }
    return true;
  }

private:
  MyASTVisitor Visitor;
  CompilerInstance& CI;
//   DynamicIDHandler& handler;
};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {
  }
  void EndSourceFileAction() override {
    SourceManager &SM = TheRewriter.getSourceMgr();
    llvm::errs() << "** EndSourceFileAction for: "
                 << SM.getFileEntryForID(SM.getMainFileID())->getName() << "\n";

    // Now emit the rewritten buffer.
    TheRewriter.getEditBuffer(SM.getMainFileID()).write(llvm::outs());
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    llvm::errs() << "** Creating AST consumer for: " << file << "\n";
    // handler = std::make_shared<DynamicIDHandler>();
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<MyASTConsumer>(TheRewriter, CI);//, *handler.get());
  }

private:
  Rewriter TheRewriter;
  std::shared_ptr<DynamicIDHandler> handler;
};

LangOptions getFormattingLangOpts(bool Cpp03 = false) {
  LangOptions LangOpts;
  LangOpts.CPlusPlus = 1;
  LangOpts.CPlusPlus11 = Cpp03 ? 0 : 1;
  LangOpts.CPlusPlus14 = Cpp03 ? 0 : 1;
  LangOpts.LineComment = 1;
  LangOpts.Bool = 1;
  return LangOpts;
}

int main(int argc, const char **argv) {
//   CommonOptionsParser op(argc, argv, ToolingSampleCategory);
//   ClangTool Tool(op.getCompilations(), op.getSourcePathList());

//   // ClangTool::run accepts a FrontendActionFactory, which is then used to
//   // create new objects implementing the FrontendAction interface. Here we use
//   // the helper newFrontendActionFactory to create a default factory that will
//   // return a new MyFrontendAction object every time.
//   // To further customize this, we could create our own factory class.
//   return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());

  CompilerInstance ci;
  ci.getLangOpts() = getFormattingLangOpts(false);
  DiagnosticOptions diagnosticOptions;
  ci.createDiagnostics();

  std::shared_ptr<clang::TargetOptions> pto = std::make_shared<clang::TargetOptions>();
  pto->Triple = llvm::sys::getDefaultTargetTriple();

  TargetInfo *pti = TargetInfo::CreateTargetInfo(ci.getDiagnostics(), pto);

  ci.setTarget(pti);
  ci.createFileManager();
  ci.createSourceManager(ci.getFileManager());
  ci.createPreprocessor(clang::TU_Complete);
//   ci.getPreprocessorOpts().UsePredefines = false;
  ci.createASTContext();
  SourceManager &SourceMgr = ci.getSourceManager();

  Rewriter TheRewriter;
  TheRewriter.setSourceMgr(SourceMgr, ci.getLangOpts());
  
  ci.setASTConsumer(
      llvm::make_unique<MyASTConsumer>(TheRewriter, ci)
      );//, argv[1]));

  ci.createSema(TU_Complete, nullptr);
  auto &sema = ci.getSema();
  sema.Initialize();
  DynamicIDHandler handler(&sema);
  sema.addExternalSource(&handler);

  const FileEntry *pFile = ci.getFileManager().getFile(argv[1]);
  ci.getSourceManager().setMainFileID(ci.getSourceManager().createFileID(
      pFile, clang::SourceLocation(), clang::SrcMgr::C_User));
  ci.getDiagnosticClient().BeginSourceFile(ci.getLangOpts(),
                                           &ci.getPreprocessor());
  clang::ParseAST(sema,true,false);
  ci.getDiagnosticClient().EndSourceFile();
}
