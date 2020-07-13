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

      auto type = PP.getSpelling(TypeToken);
      auto var = PP.getSpelling(IdentToken);

      function_prototype += type + " " + var;

      program_arg_types.push_back(type);
      program_parameters.push_back(var);

      auto parm_var_decl = cast<ParmVarDecl>(decl);

      if (parm_var_decl &&
          QualType::getAsString(parm_var_decl->getType().split(),
                                PrintingPolicy{{}}) ==
              "class xacc::internal_compiler::qreg") {
        bufferNames.push_back(ident->getName().str());
      }
    }
    function_prototype += ")";

    // Get Tokens as a string, rewrite code
    // with XACC api calls

    auto new_src = qcor::run_token_collector(PP, Toks, bufferNames);

    OS << function_prototype << "{\n";

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
      OS << "std::cout << \"execing: \" << quantum::getProgram()->toString() "
            "<< \"\\n\";\n";

      OS << "quantum::submit(buffers," << bufferNames.size();
    } else {
      OS << "quantum::submit(" << bufferNames[0] << ".results()";
    }

    OS << ");\n";
    OS << "}";
    OS << "\n}\n";

    OS << "class " << kernel_name << "{\n";
    OS << "public:\n";
    OS << "static void adjoint(";
    for (int i = 0; i < program_arg_types.size(); i++) {
      if (i > 0) {
        OS << ",";
      }
      auto arg_type = program_arg_types[i];
      auto arg_var = program_parameters[i];

      OS << arg_type << " " << arg_var;
    }
    OS << ") {\n";

    OS << "quantum::initialize(\"" << qpu_name << "\", \"" << kernel_name
       << "\");\n";
    for (auto &buf : bufferNames) {
      OS << buf << ".setNameAndStore(\"" + buf + "\");\n";
    }

    if (shots > 0) {
      OS << "quantum::set_shots(" << shots << ");\n";
    }

    OS << new_src;

    OS << "quantum::adjoint();\n";

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
    OS << "}\n";

    // close adjoint()
    OS << "}\n";
    // close class
    OS << "};";

    auto s = OS.str();
    qcor::info("[qcor syntax-handler] Rewriting " + kernel_name + " to\n\n" +
               s);
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

//// =================== COMMON CODE ==============/////
// Base class of all kernels:
// This will handle Adjoint() and Control() in an AUTOMATIC way,
// i.e. not taking into account if it is self-adjoint.
// Technically, we can define sub-classes for those special cases
// and then allow users to annotate kernels as self-adjoint for instance.
// class KernelBase {
// public:
//   KernelBase(xacc::internal_compiler::qreg q,
//              std::shared_ptr<xacc::CompositeInstruction> bodyComposite)
//       : m_qreg(q), m_usedAsCallable(true), m_body(bodyComposite) {}
//   // Adjoint
//   virtual KernelBase adjoint() {
//     // Copy this
//     KernelBase adjointKernel(this, "KernelName_ADJ");
//     // Reverse all instructions in m_body and replace instructions
//     // with their adjoint:
//     // T -> Tdag; Rx(theta) -> Rx(-theta), etc.
//     auto instructions = adjointKernel.m_body->getInstructions();
//     // Assert that we don't have measurement
//     if (!std::all_of(
//             instructions.cbegin(), instructions.cend(),
//             [](const auto &inst) { return inst->name() != "Measure"; })) {
//       xacc::error(
//           "Unable to create Adjoint for kernels that have Measure operations.");
//     }
//     std::reverse(instructions.begin(), instructions.end());
//     for (const auto &inst : instructions) {
//       // Parametric gates:
//       if (inst->name() == "Rx" || inst->name() == "Ry" ||
//           inst->name() == "Rz" || inst->name() == "CPHASE") {
//         inst->setParameter(0, -inst->getParameter(0).as<double>());
//       }
//       // TODO: Handles T and S gates, etc... => T -> Tdg
//     }
//     adjointKernel.m_body->clear();
//     adjointKernel.m_body->addInstructions(instructions);
//     return adjointKernel;
//   }
//   virtual KernelBase ctrl(size_t ctrlIdx) {
//     // Copy this
//     KernelBase controlledKernel(this, "KernelName_CTRL");
//     // Use the controlled gate module of XACC to transform
//     // controlledKernel.m_body
//     auto ctrlKernel = quantum::controlledKernel(m_body, ctrlIdx);
//     // Set the body of the returned kernel instance.
//     controlledKernel.m_body = ctrlKernel;
//     return controlledKernel;
//   }
//   // Destructor:
//   // called right after the object invocation:
//   // e.g.
//   // Case 1: free-standing invocation:
//   //  ... code ...
//   // kernelFuncClass(abc);
//   // -> DTor called here
//   // ... code ...
//   // Cade 2: chaining
//   //  ... code ...
//   // kernelFuncClass(abc).adjoint();
//   // -> DTor of the Adjoint instance called here (m_usedAsCallable = true)
//   // hence adding the adjoint body to the global composite.
//   // -> DTor of the kernelFuncClass(abc) instance called here (m_usedAsCallable
//   // = false) hence having no effect.
//   // ... code ...
//   virtual ~KernelBase() {
//     // This is used as a CALLABLE
//     if (m_usedAsCallable) {
//       // Add all instructions to the global program.
//       quantum::program->addInstructions(m_body->getInstructions());
//     }
//   }
//   // Default move CTor
//   KernelBase(KernelBase &&) = default;

// protected:
//   // Copy ctor:
//   // Deep copy of the CompositeInstruction to prevent dangling references.
//   KernelBase(KernelBase *other, const std::string &in_optional_newName = "") {
//     const auto kernelName = in_optional_newName.empty() ? other->m_body->name()
//                                                         : in_optional_newName;
//     auto provider = xacc::getIRProvider("quantum");
//     m_body = provider->createComposite(kernelName);
//     for (const auto &inst : other->m_body->getInstructions()) {
//       m_body->addInstruction(inst->clone());
//     }
//     m_qreg = other->m_qreg;
//     m_usedAsCallable = true;
//     // The copied kernel becomes *INACTIVE*
//     other->m_usedAsCallable = false;
//   }
//   // Denote if this instance was use as a *Callable*
//   // i.e.
//   // kernelFuncClass(qubitReg); => TRUE
//   // kernelFuncClass(qubitReg).adjoint(); => FALSE (on the original
//   // kernelFuncClass instance) but TRUE for the one returned by the adjoint()
//   // member function. This will allow arbitrary chaining: e.g.
//   // kernelFuncClass(qubitReg).adjoint().ctrl(k); only the last kernel returned
//   // by ctrl() will be the *Callable*;
//   bool m_usedAsCallable;
//   // The XACC composite instruction described by this kernel body:
//   std::shared_ptr<xacc::CompositeInstruction> m_body;
//   // From kernel params:
//   xacc::internal_compiler::qreg m_qreg;
// };
// //// =================== END COMMON CODE ==============/////
// // The above code can be placed in a header file which is then injected.
// /// ============= ORIGINAL CODE ===================
// // Assume we are rewriting this:
// // __qpu__ void kernelFunc(qreg q, double angle) {
// //   H(q[0]);
// //   CNOT(q[0], q[1]);
// //   Rx(q[0], angle);
// // }
// /// =============  CODE GEN ======================////
// // kernel function: returns the class object.
// KernelBase kernelFunc(xacc::internal_compiler::qreg q, double angle) {
//   quantum::initialize("qpp", "KERNEL_NAME");
//   q.setNameAndStore("q");
//   auto provider = xacc::getIRProvider("quantum");
//   // Kernel name (function name)
//   // BODY to denote it's the original body
//   auto kernelBody = provider->createComposite("KernelName_BODY");
//   // Rewrite from function body:
//   // TODO: QRT functions to take an composite instruction arg
//   // hence added instructions to that composite.
//   // HACK: for testing, swapping the *GLOBAL* program.
//   auto cachedGlobalProgram = quantum::program;
//   // Set the program to this body composite,
//   // hence we can listen to all the QRT instructions below.
//   quantum::program = kernelBody;
//   // ======== QRT Code ===========
//   // Rewrite from the __qpu__ body
//   quantum::h(q[0]); // Ideally, we'll do quantum::h(q[0], kernelBody);
//   quantum::cnot(q[0], q[1]);
//   quantum::rx(q[0], angle);
//   // ======== QRT Code ===========
//   // Restore the global program
//   quantum::program = cachedGlobalProgram;
//   KernelBase instance(q, kernelBody);
//   return instance;
// }
// /// =============  END CODE GEN ======================////
// // Note: The most difficult part is to *CHANGE* the function signature from
// // returning *void* to returing *KernelBase*,
// // i.e. we need to be able to rewrite:
// // "__qpu__ void" ==> "__qpu__ KernelBase" (__qpu__ is handled by the
// // pre-processor) Possibility: the *qcor* script to do that before calling clang
// // ??
// //////////////////////////////////////////////////
// // TEST kernel-in-kernel
// // __qpu__ void nestedFunc(qreg q, double angle1, double angle2) {
// //   kernelFunc(q, angle1).adjoint();
// //   Ry(q[1], angle2);
// //   Measure(q[0]);
// // }
// /// =============  CODE GEN ======================////
// KernelBase nestedFunc(xacc::internal_compiler::qreg q, double angle1,
//                       double angle2) {
//   quantum::initialize("qpp", "KERNEL_NAME");
//   q.setNameAndStore("q");
//   auto provider = xacc::getIRProvider("quantum");
//   // Kernel name (function name)
//   // BODY to denote it's the original body
//   auto kernelBody = provider->createComposite("KernelName_BODY");
//   // Rewrite from function body:
//   auto cachedGlobalProgram = quantum::program;
//   // Set the program to this body composite,
//   // hence we can listen to all the QRT instructions below.
//   quantum::program = kernelBody;
//   // ======== QRT Code ===========
//   // Call other kernels (i.e. left unchanged)
//   // Support arbitrary chaining here as well.
//   kernelFunc(q, angle1).adjoint();
//   // Some more gates:
//   quantum::ry(q[1], angle2);
//   quantum::mz(q[0]);
//   // ======== QRT Code ===========
//   // Restore the global program
//   quantum::program = cachedGlobalProgram;
//   KernelBase instance(q, kernelBody);
//   return instance;
// }
// /// ============= END CODE GEN ======================////
// // Classical code:
// int main(int argc, char **argv) {
//   // Allocate 3 qubits
//   auto q = qalloc(3);
//   // Can try any of the following things:
//   // kernelFunc(q, 1.234);
//   // kernelFunc(q, 1.234).adjoint();
//   // kernelFunc(q, 1.234).ctrl(2);
//   // I'm crazy :)
//   // kernelFunc(q, 1.234).adjoint().ctrl(2).adjoint();
//   // Nested case:
//   // Note: we cannot `adjoint` or `control` the nestedFunc
//   // since it contains Measure (throw).
//   nestedFunc(q, 1.23, 4.56);
//   // This should include instructions from the above kernel,
//   // which is added when the Dtor is called.
//   std::cout << "Program: \n" << quantum::program->toString() << "\n";
//   // dump the results
//   q.print();
// }
} // namespace

static SyntaxHandlerRegistry::Add<QCORSyntaxHandler>
    X("qcor", "qcor quantum kernel syntax handler");

static FrontendPluginRegistry::Add<QCORArgs> XX("qcor-args", "");
