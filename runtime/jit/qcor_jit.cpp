#include "qcor_jit.hpp"
#include "qcor_syntax_handler.hpp"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/MacroArgs.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/ModuleLoader.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include <clang/CodeGen/CodeGenAction.h>

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <cxxabi.h>
#include <memory>

#include "qcor_clang_wrapper.hpp"
#include "xacc_internal_compiler.hpp"

using namespace llvm;
using namespace llvm::orc;

#include <iostream>
#include <regex>

namespace qcor {
using namespace clang;

class LexerHelper {
protected:
  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  LangOptions LangOpts;
  std::shared_ptr<clang::TargetOptions> TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> Target;

public:
  LexerHelper()
      : FileMgr(FileMgrOpts), DiagID(new DiagnosticIDs()),
        Diags(DiagID, new DiagnosticOptions, new IgnoringDiagConsumer()),
        SourceMgr(Diags, FileMgr), TargetOpts(new clang::TargetOptions) {
    TargetOpts->Triple = "x86_64-apple-darwin11.1.0";
    Target = TargetInfo::CreateTargetInfo(Diags, TargetOpts);
  }

  std::unique_ptr<Preprocessor> CreatePP(StringRef Source,
                                         TrivialModuleLoader &ModLoader) {
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
  Lex(StringRef Source) {
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
};

template <class Op> void split(const std::string &s, char delim, Op op) {
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, delim);) {
    *op++ = item;
  }
}

//------------------------------------------------------------------------------
// Name: split
//------------------------------------------------------------------------------
inline std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}

void ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                  [](int ch) { return !std::isspace(ch); }));
}

// trim from end (in place)
void rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](int ch) { return !std::isspace(ch); })
              .base(),
          s.end());
}

// trim from both ends (in place)
void trim(std::string &s) {
  ltrim(s);
  rtrim(s);
}

const std::pair<std::string, std::string>
QJIT::run_syntax_handler(const std::string &kernel_src) {
  // first do some analysis on the string to pick out
  // from the function prototype the arg_types, arg_vars, and
  // bufferNames
  int i, size = 0;
  for (i = kernel_src.find_first_of("(") - 1;; i--) {
    if (kernel_src[i] == ' ')
      break;

    size++;
  }

  auto kernel_name = kernel_src.substr(i, size + 1);
  trim(kernel_name);
  auto args_signature = kernel_src.substr(
      kernel_src.find_first_of("("),
      kernel_src.find_first_of("{") - kernel_src.find_first_of("(") - 1);

  // Remove the parentheses
  args_signature.erase(
      std::remove_if(args_signature.begin(), args_signature.end(),
                     [](char ch) { return ch == '(' || ch == ')'; }),
      args_signature.end());

  std::vector<std::string> arg_types, arg_vars, bufferNames;
  auto args_split = split(args_signature, ',');
  for (auto &arg : args_split) {
    auto arg_var = split(arg, ' ');
    if (arg_var[0] == "qreg") {
      bufferNames.push_back(arg_var[1]);
    }
    arg_types.push_back(arg_var[0]);
    arg_vars.push_back(arg_var[1]);
  }

  // second, lex the kernel_src
  std::string temp_kernel_src = kernel_src;
  std::string function_body = kernel_src.substr(
      kernel_src.find_first_of("{") + 1,
      kernel_src.find_last_of("}") - kernel_src.find_first_of("{") - 1);

  LexerHelper helper;
  auto [tokens, PP] = helper.Lex(function_body);
  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::string Replacement;
  llvm::raw_string_ostream ReplacementOS(Replacement);

  QCORSyntaxHandler handler;
  handler.GetReplacement(*PP.get(), kernel_name, arg_types, arg_vars,
                         bufferNames, cached, ReplacementOS);
  ReplacementOS.flush();

  std::string preamble =
      "void " + kernel_name + "(" + arg_types[0] + " " + arg_vars[0];
  for (int j = 1; j < arg_types.size(); j++) {
    preamble += ", " + arg_types[j] + " " + arg_vars[j];
  }

  return std::make_pair(
      kernel_name,
      preamble + ") {\n" + Replacement +
          "\n// Fix for __dso_handle symbol not found\nint __dso_handle = 1;\n");
}
QJIT::~QJIT() {}

class LLVMJIT {
private:
  ExecutionSession ES;
  RTDyldObjectLinkingLayer ObjectLayer;
  IRCompileLayer CompileLayer;

  DataLayout DL;
  MangleAndInterner Mangle;
  ThreadSafeContext Ctx;

  JITDylib &MainJD;

public:
  LLVMJIT(JITTargetMachineBuilder JTMB, DataLayout DL)
      : ObjectLayer(ES,
                    []() { return std::make_unique<SectionMemoryManager>(); }),
        CompileLayer(ES, ObjectLayer, ConcurrentIRCompiler(std::move(JTMB))),
        DL(std::move(DL)), Mangle(ES, this->DL),
        Ctx(std::make_unique<LLVMContext>()),
        MainJD(ES.createJITDylib("<main>")) {
    MainJD.addGenerator(
        cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
            DL.getGlobalPrefix())));
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  static Expected<std::unique_ptr<LLVMJIT>> Create() {
    auto JTMB = JITTargetMachineBuilder::detectHost();

    if (!JTMB)
      return JTMB.takeError();

    auto DL = JTMB->getDefaultDataLayoutForTarget();
    if (!DL)
      return DL.takeError();

    return std::make_unique<LLVMJIT>(std::move(*JTMB), std::move(*DL));
  }

  const DataLayout &getDataLayout() const { return DL; }

  LLVMContext &getContext() { return *Ctx.getContext(); }

  Error addModule(std::unique_ptr<llvm::Module> M) {

    // FIXME hook up to cmake
    MainJD.addGenerator(cantFail(DynamicLibrarySearchGenerator::Load(
        "/home/cades/.xacc/lib/libxacc.so", DL.getGlobalPrefix())));
    MainJD.addGenerator(cantFail(DynamicLibrarySearchGenerator::Load(
        "/home/cades/.xacc/lib/libqrt.so", DL.getGlobalPrefix())));
    MainJD.addGenerator(cantFail(DynamicLibrarySearchGenerator::Load(
        "/home/cades/.xacc/lib/libqcor.so", DL.getGlobalPrefix())));
    MainJD.addGenerator(cantFail(DynamicLibrarySearchGenerator::Load(
        "/home/cades/.xacc/lib/libCppMicroServices.so", DL.getGlobalPrefix())));

    return CompileLayer.add(MainJD, ThreadSafeModule(std::move(M), Ctx));
  }

  Expected<JITEvaluatedSymbol> lookup(StringRef Name) {
    return ES.lookup({&MainJD}, Mangle(Name.str()));
  }
};

QJIT::QJIT() {}

void QJIT::jit_compile(const std::string &code) {
  auto [kernel_name, new_code] = run_syntax_handler(code);

  // FIXME Hook up some form of caching...
  auto act = qcor::emit_llvm_ir(new_code);
  module = act->takeModule();
  
  auto demangle = [](const char *name) {
    int status = -1;
    std::unique_ptr<char, void (*)(void *)> res{
        abi::__cxa_demangle(name, NULL, NULL, &status), std::free};
    return (status == 0) ? res.get() : std::string(name);
  };

  // FIXME Get the kernel(composite, args...) function too
  std::string mangled_name = "";
  for (Function &f : *module) {
    auto name = f.getName().str();
    if (demangle(name.c_str()).find(kernel_name) != std::string::npos) {
      mangled_name = name;
      break;
    }
  }

  if (!jit) {
    jit = cantFail(qcor::LLVMJIT::Create());
  }

  auto error = jit->addModule(std::move(module));
  if (error) {
    // errs() << "adding mod error\n";
  }

  auto symbol = cantFail(jit->lookup(mangled_name));
  auto rawFPtr = symbol.getAddress();
  kernel_name_to_f_ptr.insert({kernel_name, rawFPtr});

  return;
}

} // namespace qcor
