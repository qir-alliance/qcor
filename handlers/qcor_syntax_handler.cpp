#include "token_collector_util.hpp"
#include <iostream>
#include <regex>
#include <sstream>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Type.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Parse/Parser.h"

using namespace clang;

namespace {

bool qrt = false;
std::string qpu_name = "qpp";
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

    // auto src_txt = Lexer::getSourceText(
    //     CharSourceRange::getTokenRange(SourceRange(
    //         Toks[0].getLocation(), Toks[Toks.size() - 1].getLocation())),
    //     sm, lo);


    // Get the Function Type Info from the Declarator,
    // If the function has no arguments, then we throw an error
    const DeclaratorChunk::FunctionTypeInfo &FTI = D.getFunctionTypeInfo();
    std::string kernel_name = D.getName().Identifier->getName().str();

    std::string function_prototype = "void " + kernel_name + "(";
    // Loop over the function arguments and get the
    // buffer name and any program parameter doubles.
    std::vector<std::string> program_parameters, program_arg_types;
    std::vector<std::string> bufferNames;
    for (unsigned int ii = 0; ii < FTI.NumParams; ii++) {
      if (ii > 0) {
        function_prototype += ", ";
      }

      auto &paramInfo = FTI.Params[ii];
      Token IdentToken, TypeToken;
      auto ident = paramInfo.Ident;
      auto &decl = paramInfo.Param;
      PP.getRawToken(paramInfo.IdentLoc, IdentToken);
      PP.getRawToken(decl->getBeginLoc(), TypeToken);

      function_prototype +=
          PP.getSpelling(TypeToken) + " " + PP.getSpelling(IdentToken);

      auto parm_var_decl = cast<ParmVarDecl>(decl);
      if (parm_var_decl) {
        auto type = QualType::getAsString(parm_var_decl->getType().split(),
                                          PrintingPolicy{{}});
        program_arg_types.push_back(type);
        program_parameters.push_back(ident->getName().str());
        if (type == "class xacc::internal_compiler::qreg") {
          bufferNames.push_back(ident->getName().str());
        }
      }
    }
    function_prototype += ")";

    // Get Tokens as a string, rewrite code
    // with XACC api calls

    auto new_src = qcor::run_token_collector(PP, Toks, bufferNames, function_prototype );

    OS << "quantum::initialize(\"" << qpu_name << "\", \"" << kernel_name
       << "\");\n";
    for (auto &buf : bufferNames) {
      OS << buf << ".setNameAndStore(\"" + buf + "\");\n";
    }

    if (shots > 0) {
      OS << "quantum::set_shots(" << shots << ");\n";
    }
    OS << new_src;
    OS << "if (__execute) {\n";

    if (bufferNames.size() > 1) {
      OS << "xacc::AcceleratorBuffer * buffers[" << bufferNames.size()
         << "] = {";
      OS << bufferNames[0] << ".results()";
      for (unsigned int k = 1; k < bufferNames.size(); k++) {
        OS << ", " << bufferNames[k] << ".results()";
      }
      OS << "};\n";
      OS << "quantum::submit(buffers," << bufferNames.size();
    } else {
      OS << "quantum::submit(" << bufferNames[0] << ".results()";
    }

    OS << ");\n";
    OS << "}";

    auto s = OS.str();
    qcor::info("[qcor syntax-handler] Rewriting " + kernel_name + " to\n\n" +
               function_prototype + "{\n" + s.substr(2, s.length()) + "\n}");
  }

  void AddToPredefines(llvm::raw_string_ostream &OS) override {
    OS << "#include \"qrt.hpp\"\n";

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
