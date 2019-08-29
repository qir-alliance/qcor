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

#include "XACC.hpp"
#include "qcor_ast_consumer.hpp"
#include "xacc_service.hpp"

using namespace clang;

class QCORFrontendAction : public clang::ASTFrontendAction {

public:
  QCORFrontendAction(Rewriter &rw, const std::string file)
      : rewriter(rw), fileName(file) {}

protected:
  Rewriter &rewriter;
  std::string fileName;
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef /* dummy */) override {
    return std::make_unique<qcor::compiler::QCORASTConsumer>(Compiler,
                                                             rewriter);
  }

  void ExecuteAction() override {
    CompilerInstance &CI = getCompilerInstance();
    CI.createSema(getTranslationUnitKind(), nullptr);
    rewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());

    auto fuzzyParser =
        std::make_shared<qcor::compiler::FuzzyParsingExternalSemaSource>();
    fuzzyParser->initialize();
    CI.getSema().addExternalSource(fuzzyParser.get());

    // FIXME Hook this back up
    // auto pragmaHandlers =
    // xacc::getServices<qcor::compiler::QCORPragmaHandler>(); for (auto p :
    // pragmaHandlers) {
    //     p->setRewriter(&rewriter);
    //     CI.getSema().getPreprocessor().AddPragmaHandler(p.get());
    // }

    ParseAST(CI.getSema());

    // for (auto& p : pragmaHandlers) {
    //     CI.getSema().getPreprocessor().RemovePragmaHandler(p.get());
    // }

    CI.getDiagnosticClient().EndSourceFile();

    // Get the rewrite buffer
    const RewriteBuffer *RewriteBuf =
        rewriter.getRewriteBufferFor(CI.getSourceManager().getMainFileID());

    // if not null, rewrite to .fileName_out.cpp
    if (RewriteBuf) {
      auto getFileName = [](const std::string &s) {
        char sep = '/';

        size_t i = s.rfind(sep, s.length());
        if (i != std::string::npos) {
          return (s.substr(i + 1, s.length() - i));
        }

        return std::string("");
      };

      auto fileNameNoPath = getFileName(fileName);

      if (!fileNameNoPath.empty()) {
          fileName = fileNameNoPath;
      }
      
      std::string outName(fileName);
      size_t ext = outName.rfind(".");
      if (ext == std::string::npos)
        ext = outName.length();
      outName.insert(ext, "_out");
      outName = "." + outName;

      std::error_code OutErrorInfo;
      std::error_code ok;
      llvm::raw_fd_ostream outFile(llvm::StringRef(outName), OutErrorInfo,
                                   llvm::sys::fs::F_None);

      if (OutErrorInfo == ok) {
        auto s = std::string(RewriteBuf->begin(), RewriteBuf->end());
        outFile << s;
      } else {
        llvm::errs() << "Cannot open " << outName << " for writing\n";
        llvm::errs() << OutErrorInfo.message() << "\n";
      }
      outFile.close();
    } else {

      // Do we need to do anything here?
    }
  }
};

int main(int argc, char **argv) {

  xacc::Initialize(argc, argv);

  // Get filename
  std::string fileName(argv[argc - 1]);
  if (!xacc::fileExists(fileName)) {
    xacc::error("File " + fileName + " does not exist.");
  }

  std::ifstream t(fileName);
  std::string src((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());

  // Initialize rewriter
  Rewriter Rewrite;

  auto action = new QCORFrontendAction(Rewrite, fileName);
  std::vector<std::string> args{"-Wno-dangling", "-std=c++14",
                                "-I@CMAKE_INSTALL_PREFIX@/include/qcor",
                                "-I@CMAKE_INSTALL_PREFIX@/include/xacc"};

  if (!tooling::runToolOnCodeWithArgs(action, src, args)) {
    xacc::error("Error running qcor compiler.");
  }

  return 0;
}