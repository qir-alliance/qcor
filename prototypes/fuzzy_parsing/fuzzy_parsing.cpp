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
#include <iostream>
#include <sstream>
#include <string>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Parse/Parser.h"
#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Sema/ExternalSemaSource.h>
#include <clang/Sema/Lookup.h>
#include <clang/Sema/Sema.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

static llvm::cl::OptionCategory ToolingSampleCategory("Tooling Sample");

// By implementing RecursiveASTVisitor, we can specify which AST nodes
// we're interested in by overriding relevant methods.
class MyASTVisitor : public RecursiveASTVisitor<MyASTVisitor> {
public:
  MyASTVisitor(Rewriter &R, CompilerInstance &c)
      : TheRewriter(R), TheCompInst(c) {}

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

    auto xaccKernelLambdaStr = Lexer::getSourceText(
                     CharSourceRange(LE->getSourceRange(), true), SM, lo)
                     .str();
    std::cout << xaccKernelLambdaStr << "\n";

    TheRewriter.ReplaceText(LE->getBody()->getSourceRange(), "{ int hello = 0; }");
    return true;
  }

private:
  Rewriter &TheRewriter;
  CompilerInstance &TheCompInst;
};

class DynamicIDHandler : public clang::ExternalSemaSource {
public:
  DynamicIDHandler(Sema *Sema) : m_Context(Sema->getASTContext()) {}
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
    
    // FIXME Figure out how to make this check general
    if (Name.getAsString() == "X" || Name.getAsString()== "analog" || Name.getAsString() == "autogen") {
      std::cout << "Looking Up Unqualified " << Name.getAsString() << "\n";

    //   std::cout << S->isClassScope() << ", " << S->isFunctionScope() << "\n";

    //   auto fnp = S->getFnParent();
    //   fnp->dump();
    //   std::cout << "DECL " << fnp->getEntity()->getDeclKindName() << "\n";
    //   fnp->getEntity()->dumpLookups();
    
      IdentifierInfo *II = Name.getAsIdentifierInfo();
      SourceLocation Loc = R.getNameLoc();
      VarDecl *Result =
          VarDecl::Create(m_Context, R.getSema().getFunctionLevelDeclContext(),
                          Loc, Loc, II, m_Context.DependentTy,
                          0, SC_None);
      
      if (Result) {
        R.addDecl(Result);
        // Say that we can handle the situation. Clang should try to recover
        return true;
      } else {
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
  MyASTConsumer(Rewriter &R, CompilerInstance &c) : Visitor(R, c) {}

  // Override the method that gets called for each parsed top-level
  // declaration.
  bool HandleTopLevelDecl(DeclGroupRef DR) override {
    for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b) {
      // Traverse the declaration using our AST visitor.
      Visitor.TraverseDecl(*b);
    //   (*b)->dump();
    }
    return true;
  }

private:
  MyASTVisitor Visitor;
};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {}
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
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<MyASTConsumer>(TheRewriter,
                                            CI);
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

// static void SetInstallDir(SmallVectorImpl<const char *> &argv,
//                           Driver &TheDriver, bool CanonicalPrefixes) {
//   // Attempt to find the original path used to invoke the driver, to determine
//   // the installed path. We do this manually, because we want to support that
//   // path being a symlink.
//   SmallString<128> InstalledPath(argv[0]);

//   // Do a PATH lookup, if there are no directory components.
//   if (llvm::sys::path::filename(InstalledPath) == InstalledPath)
//     if (llvm::ErrorOr<std::string> Tmp = llvm::sys::findProgramByName(
//             llvm::sys::path::filename(InstalledPath.str())))
//       InstalledPath = *Tmp;

//   // FIXME: We don't actually canonicalize this, we just make it absolute.
//   if (CanonicalPrefixes)
//     llvm::sys::fs::make_absolute(InstalledPath);

//   StringRef InstalledPathParent(llvm::sys::path::parent_path(InstalledPath));
//   if (llvm::sys::fs::exists(InstalledPathParent))
//     TheDriver.setInstalledDir(InstalledPathParent);
// }

int main(int argc, const char **argv) {
  CompilerInstance ci;
  ci.getLangOpts() = getFormattingLangOpts(false);
  DiagnosticOptions diagnosticOptions;
  ci.createDiagnostics();

  std::shared_ptr<clang::TargetOptions> pto =
      std::make_shared<clang::TargetOptions>();
  pto->Triple = llvm::sys::getDefaultTargetTriple();

  TargetInfo *pti = TargetInfo::CreateTargetInfo(ci.getDiagnostics(), pto);

  ci.setTarget(pti);
  ci.createFileManager();
  ci.createSourceManager(ci.getFileManager());
  ci.createPreprocessor(clang::TU_Complete);
  ci.createASTContext();
  SourceManager &SourceMgr = ci.getSourceManager();

  Rewriter TheRewriter;
  TheRewriter.setSourceMgr(SourceMgr, ci.getLangOpts());

  ci.setASTConsumer(llvm::make_unique<MyASTConsumer>(TheRewriter, ci));

  ci.createSema(TU_Complete, nullptr);
  auto &sema = ci.getSema();
  sema.Initialize();
  DynamicIDHandler handler(&sema);
  if (argc > 2) sema.addExternalSource(&handler);

  const FileEntry *pFile = ci.getFileManager().getFile(argv[1]);
  ci.getSourceManager().setMainFileID(ci.getSourceManager().createFileID(
      pFile, clang::SourceLocation(), clang::SrcMgr::C_User));
  ci.getDiagnosticClient().BeginSourceFile(ci.getLangOpts(),
                                           &ci.getPreprocessor());
  clang::ParseAST(sema, true, false);
  ci.getDiagnosticClient().EndSourceFile();

   const RewriteBuffer *RewriteBuf =
      TheRewriter.getRewriteBufferFor(SourceMgr.getMainFileID());
  llvm::outs() << "\n" << "#include \"qcor_integration_header.hpp\"\n" << std::string(RewriteBuf->begin(), RewriteBuf->end());
  
  SmallVector<const char *, 256> argv2(argv, argv + argc);
  
  std::string Path = "/usr/bin/clang++-8";//GetExecutablePath(argv[0], CanonicalPrefixes);

  for (auto& v : argv2) {
      std::cout << "HELLO: " << v << "\n";
  }
//   IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts =
//       CreateAndPopulateDiagOpts(argv);

//   TextDiagnosticPrinter *DiagClient
//     = new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);
//   FixupDiagPrefixExeName(DiagClient, Path);

//   IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

//   DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagClient);

//   if (!DiagOpts->DiagnosticSerializationFile.empty()) {
//     auto SerializedConsumer =
//         clang::serialized_diags::create(DiagOpts->DiagnosticSerializationFile,
//                                         &*DiagOpts, /*MergeChildRecords=*/true);
//     Diags.setClient(new ChainedDiagnosticConsumer(
//         Diags.takeClient(), std::move(SerializedConsumer)));
//   }

//   ProcessWarningOptions(Diags, *DiagOpts, /*ReportDiags=*/false);

//   Driver TheDriver(Path, llvm::sys::getDefaultTargetTriple(), Diags);
//   SetInstallDir(argv, TheDriver, CanonicalPrefixes);
//   TheDriver.setTargetAndMode(TargetAndMode);

//   insertTargetAndModeArgs(TargetAndMode, argv, SavedStrings);

//   SetBackdoorDriverOutputsFromEnvVars(TheDriver);

//   std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(argv));
//   int Res = 1;
//   if (C && !C->containsError()) {
//     SmallVector<std::pair<int, const Command *>, 4> FailingCommands;
//     Res = TheDriver.ExecuteCompilation(*C, FailingCommands);

//     // Force a crash to test the diagnostics.
//     if (TheDriver.GenReproducer) {
//       Diags.Report(diag::err_drv_force_crash)
//         << !::getenv("FORCE_CLANG_DIAGNOSTICS_CRASH");

//       // Pretend that every command failed.
//       FailingCommands.clear();
//       for (const auto &J : C->getJobs())
//         if (const Command *C = dyn_cast<Command>(&J))
//           FailingCommands.push_back(std::make_pair(-1, C));
//     }

//     for (const auto &P : FailingCommands) {
//       int CommandRes = P.first;
//       const Command *FailingCommand = P.second;
//       if (!Res)
//         Res = CommandRes;

//       // If result status is < 0, then the driver command signalled an error.
//       // If result status is 70, then the driver command reported a fatal error.
//       // On Windows, abort will return an exit code of 3.  In these cases,
//       // generate additional diagnostic information if possible.
//       bool DiagnoseCrash = CommandRes < 0 || CommandRes == 70;
// #ifdef _WIN32
//       DiagnoseCrash |= CommandRes == 3;
// #endif
//       if (DiagnoseCrash) {
//         TheDriver.generateCompilationDiagnostics(*C, *FailingCommand);
//         break;
//       }
//     }
//   }

//   Diags.getClient()->finish();

//   // If any timers were active but haven't been destroyed yet, print their
//   // results now.  This happens in -disable-free mode.
//   llvm::TimerGroup::printAll(llvm::errs());

// #ifdef _WIN32
//   // Exit status should not be negative on Win32, unless abnormal termination.
//   // Once abnormal termiation was caught, negative status should not be
//   // propagated.
//   if (Res < 0)
//     Res = 1;
// #endif

  // If we have multiple failing commands, we return the result of the first
  // failing command.

}
