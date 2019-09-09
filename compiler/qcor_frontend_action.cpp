#include "qcor_frontend_action.hpp"

namespace qcor {
namespace compiler {
void QCORFrontendAction::ExecuteAction() {

  CompilerInstance &CI = getCompilerInstance();
  CI.createSema(getTranslationUnitKind(), nullptr);
  rewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());

  auto fuzzyParser =
      std::make_shared<qcor::compiler::FuzzyParsingExternalSemaSource>(CI);
  fuzzyParser->initialize();
  // fuzzyParser->setASTContext(&CI.getASTContext());
  // fuzzyParser->setFileManager(&CI.getFileManager());
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
} // namespace compiler
} // namespace qcor