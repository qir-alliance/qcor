#include "test_utils.hpp"
LexerHelper::LexerHelper()
    : FileMgr(FileMgrOpts), DiagID(new DiagnosticIDs()),
      Diags(DiagID, new DiagnosticOptions, new IgnoringDiagConsumer()),
      SourceMgr(Diags, FileMgr), TargetOpts(new clang::TargetOptions) {
  TargetOpts->Triple = "x86_64-apple-darwin11.1.0";
  Target = TargetInfo::CreateTargetInfo(Diags, TargetOpts);
}
std::unique_ptr<Preprocessor>
LexerHelper::CreatePP(StringRef Source, TrivialModuleLoader &ModLoader) {
  std::unique_ptr<llvm::MemoryBuffer> Buf =
      llvm::MemoryBuffer::getMemBuffer(Source);
  SourceMgr.setMainFileID(SourceMgr.createFileID(std::move(Buf)));

  HeaderSearch HeaderInfo(std::make_shared<HeaderSearchOptions>(), SourceMgr,
                          Diags, LangOpts, Target.get());
  std::unique_ptr<Preprocessor> PP = std::make_unique<Preprocessor>(
      std::make_shared<PreprocessorOptions>(), Diags, LangOpts, SourceMgr,
      HeaderInfo, ModLoader,
      /*IILookup =*/nullptr,
      /*OwnsHeaderSearch =*/false);
  PP->Initialize(*Target);
  PP->EnterMainSourceFile();
  return PP;
}

std::pair<std::vector<Token>, std::unique_ptr<Preprocessor>>
LexerHelper::Lex(StringRef Source) {
  TrivialModuleLoader ModLoader;
  auto PP = CreatePP(Source, ModLoader);

  std::vector<Token> toks;
  while (1) {
    Token tok;
    PP->Lex(tok);
    if (tok.is(tok::eof))
      break;
    toks.push_back(tok);
  }

  return std::make_pair(toks, std::move(PP));
}
static std::pair<std::vector<Token>, std::unique_ptr<Preprocessor>>
getTokens(std::string source) {
  LexerHelper helper;
  return helper.Lex(source);
}