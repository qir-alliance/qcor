#include "token_collector_util.hpp"
#include <iostream>
#include <sstream>

#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Parse/Parser.h"

using namespace clang;

namespace {

bool qrt = false;
std::string qpu_name = "tnqvm";
int shots = 0;

class QCORSyntaxHandler : public SyntaxHandler {
public:
  QCORSyntaxHandler() : SyntaxHandler("qcor") {}

  void GetReplacement(Preprocessor &PP, Declarator &D, CachedTokens &Toks,
                      llvm::raw_string_ostream &OS) override {

    // FIXME need way to get backend name from user/command line

    // Get the Diagnostics engine and create a few custom
    // error messgaes
    auto &diagnostics = PP.getDiagnostics();
    auto invalid_no_args = diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "Invalid quantum kernel - must provide at least a qreg argument.");
    auto invalid_arg_type =
        diagnostics.getCustomDiagID(clang::DiagnosticsEngine::Error,
                                    "Invalid quantum kernel - args can only be "
                                    "qreg, double, or std::vector<double>.");
    auto invalid_qreg_name = diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "Invalid quantum kernel - could not discover qreg variable name.");

    // Get the function prototype as a string
    SourceManager &sm = PP.getSourceManager();
    auto lo = PP.getLangOpts();
    auto function_prototype =
        Lexer::getSourceText(CharSourceRange(D.getSourceRange(), true), sm, lo)
            .str();

    // Get the Function Type Info from the Declarator,
    // If the function has no arguments, then we throw an error
    const DeclaratorChunk::FunctionTypeInfo &FTI = D.getFunctionTypeInfo();
    std::string kernel_name = D.getName().Identifier->getName().str();
    if (!FTI.Params) {
    //   diagnostics.Report(D.getBeginLoc(), invalid_no_args);
    }

    function_prototype = "(";
    // Loop over the function arguments and get the
    // buffer name and any program parameter doubles.
    std::vector<std::string> program_parameters, program_arg_types;
    std::vector<std::string> bufferNames;
    for (unsigned int ii = 0; ii < FTI.NumParams; ii++) {
      auto &paramInfo = FTI.Params[ii];

      auto ident = paramInfo.Ident;
      auto &decl = paramInfo.Param;

      auto parm_var_decl = cast<ParmVarDecl>(decl);
      if (parm_var_decl) {
        auto type = parm_var_decl->getType().getCanonicalType().getAsString();
        program_arg_types.push_back(type);
        program_parameters.push_back(ident->getName().str());
        if (type == "class xacc::internal_compiler::qreg") {
          bufferNames.push_back(ident->getName().str());
          function_prototype += "qreg " + ident->getName().str() + ", ";
        } else {
          function_prototype += type + " " + ident->getName().str() + ", ";
        }
      }
    }
    function_prototype =
        "void " + kernel_name +
        function_prototype.substr(0, function_prototype.length() - 2) + ")";

    // Get Tokens as a string, rewrite code
    // with XACC api calls

    if (qrt) {

      qcor::run_token_collector_llvm_rt(PP, Toks, function_prototype,
                                        bufferNames, kernel_name, OS, qpu_name);

    } else {
      auto kernel_src_and_compiler =
          qcor::run_token_collector(PP, Toks, function_prototype);
      auto kernel_src = kernel_src_and_compiler.first;
      auto compiler_name = kernel_src_and_compiler.second;
      // std::cout << "HELLO:\n" << kernel_src << "\n";
      // Write new source code in place of the
      // provided quantum code tokens
      if (shots > 0) {
        OS << "compiler_InitializeXACC(\"" + qpu_name + "\", " +
                  std::to_string(shots) + ");\n";
      } else {
        OS << "compiler_InitializeXACC(\"" + qpu_name + "\");\n";
      }
      for (auto &buf : bufferNames) {
        OS << buf << ".setNameAndStore(\"" + buf + "\");\n";
      }
      OS << "auto program = getCompiled(\"" << kernel_name << "\");\n";
      OS << "if (!program) {\n";
      OS << "std::string kernel_src = R\"##(" + kernel_src + ")##\";\n";
      OS << "program = compile(\"" + compiler_name +
                "\", kernel_src.c_str());\n";
      OS << "}\n";
      // OS << "optimize(program);\n";

      OS << "if (__execute) {\n";
      OS << "program->updateRuntimeArguments(" << program_parameters[0];
      for (int i = 1; i < program_parameters.size(); i++) {
        OS << ", " << program_parameters[i];
      }
      OS << ");\n";
      // OS << "internal_set_parameters(program);\n";
      if (bufferNames.size() > 1) {
        OS << "xacc::AcceleratorBuffer * buffers[" << bufferNames.size()
           << "] = {";
        OS << bufferNames[0] << ".results()";
        for (unsigned int k = 1; k < bufferNames.size(); k++) {
          OS << ", " << bufferNames[k] << ".results()";
        }
        OS << "};\n";
        OS << "execute(buffers," << bufferNames.size() << ",program";
      } else {
        OS << "execute(" << bufferNames[0] << ".results(), program";
      }
      OS << ");\n";
      OS << "}\n";
    }

    auto s = OS.str();
    qcor::info("[qcor syntax-handler] Rewriting " + kernel_name + " to\n\n" +
               function_prototype + "{\n" + s.substr(2, s.length()) + "\n}");
  }

  void AddToPredefines(llvm::raw_string_ostream &OS) override {
    if (qrt) {
      OS << "#include \"qrt.hpp\"\n";
    }
    OS << "#include \"xacc_internal_compiler.hpp\"\nusing namespace "
          "xacc::internal_compiler;\n";
  }
};

class DoNothingConsumer : public ASTConsumer {
public:
  bool HandleTopLevelDecl(DeclGroupRef DG) override { return true; }
};

class QCORArgs : public PluginASTAction {
public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 llvm::StringRef) override {
    return std::make_unique<DoNothingConsumer>();
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      // Example error handling.
      DiagnosticsEngine &D = CI.getDiagnostics();
      if (args[i] == "-qpu") {
        if (i + 1 >= e) {
          D.Report(D.getCustomDiagID(DiagnosticsEngine::Error,
                                     "missing -qpu argument"));
          return false;
        }
        ++i;
        qpu_name = args[i];
      } else if (args[i] == "-shots") {
        if (i + 1 >= e) {
          D.Report(D.getCustomDiagID(DiagnosticsEngine::Error,
                                     "missing -shots argument"));
          return false;
        }
        ++i;
        shots = std::stoi(args[i]);
      } else if (args[i] == "-qcor-verbose") {
        qcor::set_verbose(true);
      } else if (args[i] == "-qrt") {
        qrt = true;
      }
    }
    return true;
  }

  PluginASTAction::ActionType getActionType() override {
    return AddBeforeMainAction;
  }
};

} // namespace

static SyntaxHandlerRegistry::Add<QCORSyntaxHandler>
    X("qcor", "qcor quantum kernel syntax handler");

static FrontendPluginRegistry::Add<QCORArgs> XX("qcor-args", "");
