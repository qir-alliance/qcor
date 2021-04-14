#include <clang/CodeGen/CodeGenAction.h>

#include <fstream>
#include <memory>

#include "Accelerator.hpp"
#include "CompositeInstruction.hpp"
#include "Utils.hpp"
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
#include "heterogeneous.hpp"
#include "json.hpp"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
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
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "qcor_clang_wrapper.hpp"
#include "qcor_jit.hpp"
#include "qcor_syntax_handler.hpp"
#include "qrt.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"

using namespace llvm;
using namespace llvm::orc;

#include <sys/stat.h>

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
      : FileMgr(FileMgrOpts),
        DiagID(new DiagnosticIDs()),
        Diags(DiagID, new DiagnosticOptions, new IgnoringDiagConsumer()),
        SourceMgr(Diags, FileMgr),
        TargetOpts(new clang::TargetOptions) {
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

  std::pair<std::vector<Token>, std::unique_ptr<Preprocessor>> Lex(
      StringRef Source) {
    TrivialModuleLoader ModLoader;
    auto PP = CreatePP(Source, ModLoader);

    std::vector<Token> toks;
    while (1) {
      Token tok;
      PP->Lex(tok);
      if (tok.is(tok::eof)) break;
      toks.push_back(tok);
    }

    return std::make_pair(toks, std::move(PP));
  }
};

template <class Op>
void split(const std::string &s, char delim, Op op) {
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

// Split argument signature:
// Handle templated types as well,
// e.g. "qreg q, KernelSignature<qreg,int,double> call_var"
// KernelSignature<qreg,int,double> is considered as one term (arg_type)
// Strategy: using balancing rule to match the '<' and '>' and skipping ',' delimiter.
inline std::vector<std::string> split_args_signature(const std::string &source) {
  std::vector<std::string> elems;
  std::string token;
  const int N = source.length();
  // Track a scope block, e.g. b/w '<' and '>' where comma separator is ignored
  // considered as one token. 
  std::stack<int> ignored_scopes; 
  for (int i = 0; i < N; i++) {
    const auto currentChar = source[i];
    
    // See a ',' and the stack is empty (balanced)
    if (currentChar == ',' && ignored_scopes.empty()) {
      elems.emplace_back(token);
      token.clear();
      continue;
    }

    // Open scope:
    // Note: we continue to append these characters to the current token.
    if (currentChar == '<') {
      ignored_scopes.push(i);
    }

    // Close scope
    if (currentChar == '>') {
      ignored_scopes.pop();
    }
    
    // Just add the character to the current token.
    token += currentChar;
  }

  // last token
  assert(!token.empty());
  elems.emplace_back(token);

  // for (const auto &el : elems) {
  //   std::cout << el << "\n";
  // }
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

const std::pair<std::string, std::string> QJIT::run_syntax_handler(
    const std::string &kernel_src, const bool add_het_map_kernel_ctor) {
  bool has_qpu_function_proto = kernel_src.find("__qpu__") != std::string::npos;

  // first do some analysis on the string to pick out
  // from the function prototype the arg_types, arg_vars, and
  // bufferNames
  int i, size = 0;
  for (i = kernel_src.find_first_of("(") - 1;; i--) {
    if (kernel_src[i] == ' ') break;

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
  auto args_split = split_args_signature(args_signature);
  for (auto &arg : args_split) {
    auto arg_var = split(arg, ' ');
    if (arg_var[0] == "qreg") {
      bufferNames.push_back(arg_var[1]);
    } else if (arg_var[0] == "qubit") {
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

  // std::cout << "QJIT FBODY:\n" << function_body << "\n";
  LexerHelper helper;
  auto [tokens, PP] = helper.Lex(function_body);
  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::string Replacement;
  llvm::raw_string_ostream ReplacementOS(Replacement);

  QCORSyntaxHandler handler;
  qcor::qpu_name = xacc::internal_compiler::get_qpu()->name();
  qcor::shots = quantum::get_shots();

  handler.GetReplacement(*PP.get(), kernel_name, arg_types, arg_vars,
                         bufferNames, cached, ReplacementOS,
                         add_het_map_kernel_ctor);
  ReplacementOS.flush();

  std::string preamble =
      "void " + kernel_name + "(" + arg_types[0] + " " + arg_vars[0];
  for (int j = 1; j < arg_types.size(); j++) {
    preamble += ", " + arg_types[j] + " " + arg_vars[j];
  }

  const std::string fix_dso_str = R"(
// Fix for __dso_handle symbol not found
#ifndef __FIX__DSO__HANDLE__
#define __FIX__DSO__HANDLE__ 
int __dso_handle = 1;
#endif
)";

  // std::cout << "Replacement:\n" << Replacement << "\n";
  return std::make_pair(kernel_name, Replacement + "\n" + fix_dso_str + "\n");
}

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
  LLVMJIT(JITTargetMachineBuilder JTMB, DataLayout DL,
          std::unique_ptr<LLVMContext> ctx = std::make_unique<LLVMContext>())
      : ObjectLayer(ES,
                    []() { return std::make_unique<SectionMemoryManager>(); }),
        CompileLayer(ES, ObjectLayer,
                     std::make_unique<ConcurrentIRCompiler>(std::move(JTMB))),
        DL(std::move(DL)),
        Mangle(ES, this->DL),
        Ctx(std::move(ctx)),
        MainJD(ES.createBareJITDylib("<main>")) {
    MainJD.addGenerator(
        cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
            DL.getGlobalPrefix())));
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  static Expected<std::unique_ptr<LLVMJIT>> Create() {
    auto JTMB = JITTargetMachineBuilder::detectHost();

    if (!JTMB) return JTMB.takeError();

    auto DL = JTMB->getDefaultDataLayoutForTarget();
    if (!DL) return DL.takeError();

    return std::make_unique<LLVMJIT>(std::move(*JTMB), std::move(*DL));
  }

  static Expected<std::unique_ptr<LLVMJIT>> Create(
      std::unique_ptr<LLVMContext> ctx) {
    auto JTMB = JITTargetMachineBuilder::detectHost();

    if (!JTMB) return JTMB.takeError();

    auto DL = JTMB->getDefaultDataLayoutForTarget();
    if (!DL) return DL.takeError();

    return std::make_unique<LLVMJIT>(std::move(*JTMB), std::move(*DL),
                                     std::move(ctx));
  }
  const DataLayout &getDataLayout() const { return DL; }

  LLVMContext &getContext() { return *Ctx.getContext(); }

  Error addModule(std::unique_ptr<llvm::Module> M,
                  std::vector<std::string> extra_paths = {}) {
    MainJD.addGenerator(cantFail(DynamicLibrarySearchGenerator::Load(
        "@XACC_ROOT@/lib/libxacc@CMAKE_SHARED_LIBRARY_SUFFIX@",
        DL.getGlobalPrefix())));
    MainJD.addGenerator(cantFail(DynamicLibrarySearchGenerator::Load(
        "@CMAKE_INSTALL_PREFIX@/lib/libqrt@CMAKE_SHARED_LIBRARY_SUFFIX@",
        DL.getGlobalPrefix())));
    MainJD.addGenerator(cantFail(DynamicLibrarySearchGenerator::Load(
        "@CMAKE_INSTALL_PREFIX@/lib/libqcor@CMAKE_SHARED_LIBRARY_SUFFIX@",
        DL.getGlobalPrefix())));
    MainJD.addGenerator(cantFail(DynamicLibrarySearchGenerator::Load(
        "@XACC_ROOT@/lib/libCppMicroServices@CMAKE_SHARED_LIBRARY_SUFFIX@",
        DL.getGlobalPrefix())));

    for (auto p : extra_paths) {
      MainJD.addGenerator(cantFail(DynamicLibrarySearchGenerator::Load(
          p.c_str(), DL.getGlobalPrefix())));
    }

    auto rt = MainJD.getDefaultResourceTracker();
    return CompileLayer.add(rt, ThreadSafeModule(std::move(M), Ctx));
  }

  Expected<JITEvaluatedSymbol> lookup(StringRef Name) {
    return ES.lookup({&MainJD}, Mangle(Name.str()));
  }
};

QJIT::QJIT() {
  // if tmp directory doesnt exist create it
  qjit_cache_path = std::string(std::getenv("HOME")) + "/.qjit";
  if (!xacc::directoryExists(qjit_cache_path)) {
    auto status =
        mkdir(qjit_cache_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }

  std::string cache_file_loc = qjit_cache_path + "/qjit_cache.json";
  if (!xacc::fileExists(cache_file_loc)) {
    // if it doesn't exist, create it
    std::ofstream cache(cache_file_loc);
    cache.close();
  } else {
    std::ifstream cache_file(cache_file_loc);
    std::string cache_file_contents(
        (std::istreambuf_iterator<char>(cache_file)),
        std::istreambuf_iterator<char>());
    if (!cache_file_contents.empty()) {
      auto cache_json = nlohmann::json::parse(cache_file_contents);
      auto jit_cache = cache_json["jit_cache"];
      for (auto &element : jit_cache) {
        auto key_val = element.get<std::pair<std::size_t, std::string>>();
        cached_kernel_codes.insert(key_val);
      }
    }
  }
}
void QJIT::write_cache() {
  std::string cache_file_loc = qjit_cache_path + "/qjit_cache.json";

  // MAKE SURE WE DONT OVERWRITE
  // read in any existing data, and merge
  std::ifstream cache_file(cache_file_loc);
  std::string cache_file_contents((std::istreambuf_iterator<char>(cache_file)),
                                  std::istreambuf_iterator<char>());
  if (!cache_file_contents.empty()) {
    auto cache_json = nlohmann::json::parse(cache_file_contents);
    auto jit_cache = cache_json["jit_cache"];
    std::map<std::size_t, std::string> tmp;
    for (auto &element : jit_cache) {
      auto key_val = element.get<std::pair<std::size_t, std::string>>();
      tmp.insert(key_val);
    }
    cached_kernel_codes.merge(tmp);
  }
  /////

  nlohmann::json j;
  j["jit_cache"] = cached_kernel_codes;
  auto str = j.dump();
  std::ofstream cache(cache_file_loc);
  cache << str;
  cache.close();
}
QJIT::~QJIT() { write_cache(); }

void QJIT::jit_compile(std::unique_ptr<llvm::Module> m,
                       std::vector<std::string> extra_shared_lib_paths) {
  std::vector<std::string> seen_functions;
  for (Function &f : *m) {
    auto name = f.getName().str();
    seen_functions.push_back(name);
  }

  if (!jit) {
    jit = cantFail(qcor::LLVMJIT::Create(),
                   "QJIT Error: Could not create the JIT Engine.");
  }

  // Add the Module to the JIT Engine
  cantFail(jit->addModule(std::move(m), extra_shared_lib_paths),
           "QJIT Error: Could not add the Module to the JIT Engine.");

  for (auto name : seen_functions) {
    auto symbol = cantFail(jit->lookup(name));
    auto rawFPtr = symbol.getAddress();
    kernel_name_to_f_ptr.insert({name, rawFPtr});
  }
  return;
}

void QJIT::jit_compile(const std::string &code,
                       const bool add_het_map_kernel_ctor,
                       const std::vector<std::string> &kernel_dependency,
                       const std::string &extra_functions_src,
                       std::vector<std::string> extra_headers) {
  // Run the Syntax Handler to get the kernel name and
  // the kernel code (the QuantumKernel subtype def + utility functions)
  auto [kernel_name, new_code] =
      run_syntax_handler(code, add_het_map_kernel_ctor);

  static std::unordered_map<std::string, std::string> JIT_KERNEL_RUNTIME_CACHE;
  JIT_KERNEL_RUNTIME_CACHE[kernel_name] = new_code;

  // Add dependency code if necessary:
  // Look up the previously-generated for dependency kernels and add them to
  // this kernel before compilation.
  std::string dependencyCode;
  for (const auto &dep : kernel_dependency) {
    const auto depIter = JIT_KERNEL_RUNTIME_CACHE.find(dep);
    if (depIter != JIT_KERNEL_RUNTIME_CACHE.end()) {
      dependencyCode += JIT_KERNEL_RUNTIME_CACHE[dep];
    }
  }
  // Add dependency before JIT compile:
  new_code = dependencyCode + new_code;

  // Add any extra functions to be compiled
  new_code = extra_functions_src + "\n" + new_code;

  // std::cout << "New code:\n" << new_code << "\n";
  // Hash the new code
  std::hash<std::string> hasher;
  auto hash = hasher(new_code);

  // We will use cached Modules if possible...

  std::unique_ptr<CodeGenAction> act;
  if (cached_kernel_codes.count(hash)) {
    // If we have this hash in the cache, we will grab its
    // correspoding Module bc file name and load it
    auto module_bitcode_file_name = cached_kernel_codes[hash];
    std::string full_path = qjit_cache_path + "/" + module_bitcode_file_name;

    // Load the bitcode file as Module
    SMDiagnostic error;
    auto ctx = std::make_unique<LLVMContext>();
    module = llvm::parseIRFile(full_path, error, *ctx.get());
    if (!jit) {
      // Initialize the JIT Engine
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      jit = cantFail(qcor::LLVMJIT::Create(std::move(ctx)));
    }

  } else {
    // We have not seen this code before, so we
    // need to map it to an LLVM Module
    act = qcor::emit_llvm_ir(new_code, extra_headers);
    module = act->takeModule();

    // Persist the Module to a bitcode file
    std::stringstream file_name_ss;
    file_name_ss << "__qjit_m_" << module.get() << ".bc";
    std::error_code ec;
    ToolOutputFile result(qjit_cache_path + "/" + file_name_ss.str(), ec,
                          sys::fs::F_None);
    WriteBitcodeToFile(*module, result.os());
    result.keep();

    // Add the file to the cache map, this gets persisted
    // at QJIT Destruction
    cached_kernel_codes.insert({hash, file_name_ss.str()});
  }

  // Loop over all Functions in the module
  // and get the first one that has the kernel name
  // in it as a substring. This is the corrent Function and
  // now we have it as a mangled name
  std::string mangled_name = "", hetmap_mangled_name = "",
              parent_hetmap_mangled_name = "";
  for (Function &f : *module) {
    auto name = f.getName().str();
    if (demangle(name.c_str()).find(kernel_name) != std::string::npos) {
      // First one we see is the correct kernel call
      mangled_name = name;
      break;
    }
  }

  // Insert dependency kernels as well:
  std::unordered_map<std::string, std::string> mangled_kernel_dep_map;
  for (const auto &dep : kernel_dependency) {
    std::vector<std::string> matches;
    for (Function &f : *module) {
      const auto name = f.getName().str();
      const auto demangledName = demangle(name.c_str());
      // We look for the function with the signature:
      // KernelName(shared_ptr<CompositeInstruction, Args...)
      // The problem is that there is a class named KernelName as well
      // which has a constructor with the same signature.
      // The ctor one will have a demangled name of KernelName::KernelName(...)
      // hence, we tie break them by the length.
      // Looks for a call-like symbol
      const std::string pattern = dep + "(";
      // Looks for the one that has parent Composite in the arg
      const std::string subPattern = "CompositeInstruction";
      if (demangle(name.c_str()).find(pattern) != std::string::npos &&
          demangle(name.c_str()).find(subPattern) != std::string::npos) {
        // std::cout << dep << " --> " << name << "\n";
        // std::cout << "Demangle: " << demangle(name.c_str()) << "\n";
        matches.emplace_back(name);
      }
    }
    if (matches.size() > 0) {
      // std::cout << "Matches for " << dep << ":\n";
      // for (const auto &match : matches) {
      //   std::cout << match << "\n";
      // }

      const auto chosenMatch =
          *std::min_element(matches.begin(), matches.end(),
                            [&](const std::string &s1, const std::string &s2) {
                              return demangle(s1.c_str()).length() <
                                     demangle(s2.c_str()).length();
                            });
      // std::cout << "Select match: " << chosenMatch << ": "
      //           << demangle(chosenMatch.c_str()) << "\n";
      mangled_kernel_dep_map[dep] = chosenMatch;
    }
  }

  // Find the hetmap args function
  for (Function &f : *module) {
    auto name = f.getName().str();
    if (demangle(name.c_str()).find(kernel_name + "__with_hetmap_args") !=
        std::string::npos) {
      // First one we see is the correct kernel call
      hetmap_mangled_name = name;
      break;
    }
  }

  // Find the parent composte + hetmap args function
  for (Function &f : *module) {
    auto name = f.getName().str();
    if (demangle(name.c_str())
            .find(kernel_name + "__with_parent_and_hetmap_args") !=
        std::string::npos) {
      // First one we see is the correct kernel call
      parent_hetmap_mangled_name = name;
      break;
    }
  }

  // Create the JIT Engine if we haven't already
  if (!jit) {
    jit = cantFail(qcor::LLVMJIT::Create(),
                   "QJIT Error: Could not create the JIT Engine.");
  }

  // Add the Module to the JIT Engine
  cantFail(jit->addModule(std::move(module)),
           "QJIT Error: Could not add the Module to the JIT Engine.");

  // Get the function pointer and associate it with
  // the provided kernel name
  auto symbol = cantFail(jit->lookup(mangled_name));
  auto rawFPtr = symbol.getAddress();
  kernel_name_to_f_ptr.insert({kernel_name, rawFPtr});

  for (const auto &[orig_name, mangled_name] : mangled_kernel_dep_map) {
    auto symbol = cantFail(jit->lookup(mangled_name));
    auto rawFPtr = symbol.getAddress();
    kernel_name_to_f_ptr.insert({orig_name, rawFPtr});
  }

  if (add_het_map_kernel_ctor) {
    // Get the function pointer for the hetmap kernel invocation
    auto hetmap_symbol = cantFail(jit->lookup(hetmap_mangled_name));
    auto hetmap_rawFPtr = hetmap_symbol.getAddress();
    kernel_name_to_f_ptr_hetmap.insert({kernel_name, hetmap_rawFPtr});

    // Get the function pointer for the hetmap kernel invocation with parent
    // composite
    auto parent_hetmap_symbol =
        cantFail(jit->lookup(parent_hetmap_mangled_name));
    auto parent_hetmap_rawFPtr = parent_hetmap_symbol.getAddress();
    kernel_name_to_f_ptr_parent_hetmap.insert(
        {kernel_name, parent_hetmap_rawFPtr});
  }

  return;
}

void QJIT::invoke_with_hetmap(const std::string &kernel_name,
                              xacc::HeterogeneousMap &args) {
  auto f_ptr = kernel_name_to_f_ptr_hetmap[kernel_name];
  void (*kernel_functor)(xacc::HeterogeneousMap &) =
      (void (*)(xacc::HeterogeneousMap &))f_ptr;
  kernel_functor(args);
}

std::shared_ptr<xacc::CompositeInstruction> QJIT::extract_composite_with_hetmap(
    const std::string kernel_name, xacc::HeterogeneousMap &args) {
  auto composite =
      xacc::getIRProvider("quantum")->createComposite(kernel_name + "_qjit");
  auto f_ptr = kernel_name_to_f_ptr_parent_hetmap[kernel_name];
  void (*kernel_functor)(std::shared_ptr<xacc::CompositeInstruction>,
                         xacc::HeterogeneousMap &) =
      (void (*)(std::shared_ptr<xacc::CompositeInstruction>,
                xacc::HeterogeneousMap &))f_ptr;
  kernel_functor(composite, args);
  return composite;
}

std::uint64_t QJIT::get_kernel_function_ptr(const std::string &kernelName,
                                         KernelType subType) const {
  const auto mapToUse = [&]() {
    switch (subType) {
    case KernelType::Regular:
      return kernel_name_to_f_ptr;
    case KernelType::HetMapArg:
      return kernel_name_to_f_ptr_hetmap;
    case KernelType::HetMapArgWithParent:
      return kernel_name_to_f_ptr_parent_hetmap;
    default:
      __builtin_unreachable();
    }
  }();

  const auto iter = mapToUse.find(kernelName);
  if (iter != mapToUse.cend()) {
    // std::cout << "Get pointer for '" << kernelName << "': " << std::hex
    //           << iter->second << "\n";
    return iter->second;
  } 
  
  return 0;
}
}  // namespace qcor
