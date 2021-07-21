#include "qcor_syntax_handler.hpp"

#include <iostream>
#include <regex>
#include <sstream>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Type.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "token_collector_util.hpp"

using namespace clang;

// for _QCOR_MUTEX
#include "qcor_config.hpp"

#ifdef _QCOR_MUTEX
#include <mutex>
#pragma message ("_QCOR_MUTEX is ON")
#endif

namespace qcor {
namespace __internal__developer__flags__ {
bool add_predefines = true;
}

bool qrt = false;
std::string qpu_name = "qpp";
int shots = 0;

void QCORSyntaxHandler::GetReplacement(Preprocessor &PP, Declarator &D,
                                       CachedTokens &Toks,
                                       llvm::raw_string_ostream &OS) {
  const DeclaratorChunk::FunctionTypeInfo &FTI = D.getFunctionTypeInfo();
  auto kernel_name = D.getName().Identifier->getName().str();

  std::vector<std::string> program_parameters, program_arg_types;
  std::vector<std::string> bufferNames;
  for (unsigned int ii = 0; ii < FTI.NumParams; ii++) {
    auto &paramInfo = FTI.Params[ii];
    Token IdentToken, TypeToken;
    auto ident = paramInfo.Ident;
    auto &decl = paramInfo.Param;
    auto parm_var_decl = cast<ParmVarDecl>(decl);
    PP.getRawToken(paramInfo.IdentLoc, IdentToken);
    PP.getRawToken(decl->getBeginLoc(), TypeToken);

    auto type = QualType::getAsString(parm_var_decl->getType().split(),
                                      PrintingPolicy{{}});
    // auto var = PP.getSpelling(IdentToken);
    // Using IdentifierInfo to get the correct name (after any macro expansion)
    auto var = ident->getName().str();
    if (type == "class xacc::internal_compiler::qreg") {
      bufferNames.push_back(ident->getName().str());
      type = "qreg";
    } else if (type == "qcor::qreg") {
      bufferNames.push_back(ident->getName().str());
      type = "qreg";
    } else if (type.find("xacc::internal_compiler::qubit") !=
               std::string::npos) {
      bufferNames.push_back(ident->getName().str());
      type = "qubit";
    }

    program_arg_types.push_back(type);
    program_parameters.push_back(var);

    // If this was a passed kernel as a KernelSignature, then 
    // we need to add to the kernels in translation unit
    if (auto t = dyn_cast<TypedefType>(parm_var_decl->getType())) {
      auto d = t->desugar();
      if (d.getAsString().find("KernelSignature") != std::string::npos) {
        qcor::append_kernel(var, {}, {});
      }
    }
  }

  GetReplacement(PP, kernel_name, program_arg_types, program_parameters,
                 bufferNames, Toks, OS);
}

void QCORSyntaxHandler::GetReplacement(
    Preprocessor &PP, std::string &kernel_name,
    std::vector<std::string> program_arg_types,
    std::vector<std::string> program_parameters,
    std::vector<std::string> bufferNames, CachedTokens &Toks,
    llvm::raw_string_ostream &OS, bool add_het_map_ctor) {
  
  // Add any KernelSignature or qpu_lambda to the list of known kernels.
  // Note: this GetReplacement overload is called directly from QJIT
  // vs. the standard SyntaxHandler API.
  for (int i = 0; i < program_arg_types.size(); ++i) {
    if (program_arg_types[i].find("KernelSignature") != std::string::npos ||
        program_arg_types[i].find("qcor::_qpu_lambda") != std::string::npos) {
      qcor::append_kernel(program_parameters[i], {}, {});
    }
  }
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

  // Get Tokens as a string, rewrite code
  // with XACC api calls
  qcor::append_kernel(kernel_name, program_arg_types, program_parameters);

  // for (int i = 0; i < program_arg_types.size(); i++) {
  //   if (program_arg_types[i].find("CallableKernel") != std::string::npos) {
  //     // we have a kernel we can call, need to add it to
  //     // append_kernel call.
  //     qcor::append_kernel(program_parameters[i], {}, {});
  //   }
  // }

  std::string src_to_prepend;
  auto new_src = qcor::run_token_collector(PP, Toks, src_to_prepend,
                                           kernel_name, program_arg_types,
                                           program_parameters, bufferNames);

  if (!src_to_prepend.empty()) OS << src_to_prepend;

  // Rewrite the original function
  OS << "void " << kernel_name << "(" << program_arg_types[0] << " "
     << program_parameters[0];
  for (int i = 1; i < program_arg_types.size(); i++) {
    OS << ", " << program_arg_types[i] << " " << program_parameters[i];
  }
  OS << ") {\n";

  // First re-write, forward declare a function
  // we will implement further down
  OS << "void __internal_call_function_" << kernel_name << "("
     << program_arg_types[0];
  for (int i = 1; i < program_arg_types.size(); i++) {
    OS << ", " << program_arg_types[i];
  }
  OS << ");\n";

  // Call that forward declared function with the function args
  OS << "__internal_call_function_" << kernel_name << "("
     << program_parameters[0];
  for (int i = 1; i < program_parameters.size(); i++) {
    OS << ", " << program_parameters[i];
  }
  OS << ");\n";

  // Close the transformed function
  OS << "}\n";

  // Declare the QuantumKernel subclass
  OS << "class " << kernel_name << " : public qcor::QuantumKernel<class "
     << kernel_name << ", " << program_arg_types[0];
  for (int i = 1; i < program_arg_types.size(); i++) {
    OS << ", " << program_arg_types[i];
  }
  OS << "> {\n";

  // declare the super-type as a friend
  OS << "friend class qcor::QuantumKernel<class " << kernel_name << ", "
     << program_arg_types[0];
  for (int i = 1; i < program_arg_types.size(); i++) {
    OS << ", " << program_arg_types[i];
  }
  OS << ">;\n";

  // Declare KernelSignature as a friend as well
  OS << "friend class qcor::KernelSignature<"<< program_arg_types[0];
  for (int i = 1; i < program_arg_types.size(); i++) {
    OS << ", " << program_arg_types[i];
  }
  OS << ">;\n";

  // declare protected operator()() method
  OS << "protected:\n";
#ifdef _QCOR_MUTEX
  OS << "static std::recursive_mutex& getMutex() {\n";
  OS << "static std::recursive_mutex m;\n";
  OS << "return m;\n";
  OS << "}\n";
#endif
  OS << "void operator()(" << program_arg_types[0] << " "
     << program_parameters[0];
  for (int i = 1; i < program_arg_types.size(); i++) {
    OS << ", " << program_arg_types[i] << " " << program_parameters[i];
  }
  OS << ") {\n";
#ifdef _QCOR_MUTEX
  OS << "std::lock_guard<std::recursive_mutex> lock(getMutex());\n";
#endif
  OS << "if (!parent_kernel) {\n";
  OS << "parent_kernel = "
        "qcor::__internal__::create_composite(kernel_name);\n";
  OS << "}\n";
  OS << "quantum::set_current_program(parent_kernel);\n";
  // Set the buffer in FTQC mode so that following QRT calls (in new_src) are
  // executed on that buffer.
  OS << "if (runtime_env == QrtType::FTQC) {\n";
  // We only support one buffer in FTQC mode atm.
  OS << "quantum::set_current_buffer(" << bufferNames[0] << ".results());\n";
  OS << "}\n";
  // Set the parent_kernel of this kernel to any KernelSignature
  // (even nested in container) so that even w/o parent_kernel added by the 
  // token collector, the KernelSignature will still be referring to the correct parent_kernel.
  // i.e. we can track KernelSignature coming from complex data container such as std::vector
  // rather than relying on the list of kernels in translation unit.
  OS << "init_kernel_signature_args(parent_kernel, " << program_parameters[0];
  for (int i = 1; i < program_arg_types.size(); i++) {
    OS << ", " << program_parameters[i];
  }
  OS << ");\n";
  OS << new_src << "\n";
  OS << "}\n";

  // declare public members, methods, constructors
  OS << "public:\n";
  OS << "inline static const std::string kernel_name = \"" << kernel_name
     << "\";\n";

  // First constructor, default one KERNEL_NAME(Args...);
  OS << kernel_name << "(" << program_arg_types[0] << " "
     << program_parameters[0];
  for (int i = 1; i < program_arg_types.size(); i++) {
    OS << ", " << program_arg_types[i] << " " << program_parameters[i];
  }
  OS << "): QuantumKernel<" << kernel_name << ", " << program_arg_types[0];
  for (int i = 1; i < program_arg_types.size(); i++) {
    OS << ", " << program_arg_types[i];
  }
  OS << "> (" << program_parameters[0];
  for (int i = 1; i < program_parameters.size(); i++) {
    OS << ", " << program_parameters[i];
  }
  OS << ") {}\n";

  // Second constructor, takes parent CompositeInstruction
  // KERNEL_NAME(CompositeInstruction, Args...)
  OS << kernel_name << "(std::shared_ptr<qcor::CompositeInstruction> _parent, "
     << program_arg_types[0] << " " << program_parameters[0];
  for (int i = 1; i < program_arg_types.size(); i++) {
    OS << ", " << program_arg_types[i] << " " << program_parameters[i];
  }
  OS << "): QuantumKernel<" << kernel_name << ", " << program_arg_types[0];
  for (int i = 1; i < program_arg_types.size(); i++) {
    OS << ", " << program_arg_types[i];
  }
  OS << "> (_parent, " << program_parameters[0];
  for (int i = 1; i < program_parameters.size(); i++) {
    OS << ", " << program_parameters[i];
  }
  OS << ") {}\n";

  // if (add_het_map_ctor) {
  //   // Third constructor, give us a way to provide a HeterogeneousMap of
  //   // arguments, this is used for Pythonic QJIT...
  //   // KERNEL_NAME(HeterogeneousMap args);
  //   OS << kernel_name << "(HeterogeneousMap& args): QuantumKernel<"
  //      << kernel_name << ", " << program_arg_types[0];
  //   for (int i = 1; i < program_arg_types.size(); i++) {
  //     OS << ", " << program_arg_types[i];
  //   }
  //   OS << "> (args.get<" << program_arg_types[0] << ">(\""
  //      << program_parameters[0] << "\")";
  //   for (int i = 1; i < program_parameters.size(); i++) {
  //     OS << ", "
  //        << "args.get<" << program_arg_types[i] << ">(\""
  //        << program_parameters[i] << "\")";
  //   }
  //   OS << ") {}\n";

  //   // Forth constructor, give us a way to provide a HeterogeneousMap of
  //   // arguments, and set a parent kernel - this is also used for Pythonic
  //   // QJIT... KERNEL_NAME(std::shared_ptr<CompositeInstruction> parent,
  //   // HeterogeneousMap args);
  //   OS << kernel_name
  //      << "(std::shared_ptr<CompositeInstruction> parent, HeterogeneousMap& "
  //         "args): QuantumKernel<"
  //      << kernel_name << ", " << program_arg_types[0];
  //   for (int i = 1; i < program_arg_types.size(); i++) {
  //     OS << ", " << program_arg_types[i];
  //   }
  //   OS << "> (parent, args.get<" << program_arg_types[0] << ">(\""
  //      << program_parameters[0] << "\")";
  //   for (int i = 1; i < program_parameters.size(); i++) {
  //     OS << ", "
  //        << "args.get<" << program_arg_types[i] << ">(\""
  //        << program_parameters[i] << "\")";
  //   }
  //   OS << ") {}\n";
  // }

  // Destructor definition
  OS << "virtual ~" << kernel_name << "() {\n";
  OS << "if (disable_destructor) {return;}\n";
#ifdef _QCOR_MUTEX
  OS << "std::lock_guard<std::recursive_mutex> lock(getMutex());\n";
#endif
  OS << "auto [" << program_parameters[0];
  for (int i = 1; i < program_parameters.size(); i++) {
    OS << ", " << program_parameters[i];
  }
  OS << "] = args_tuple;\n";
  OS << "operator()(" << program_parameters[0];
  for (int i = 1; i < program_parameters.size(); i++) {
    OS << ", " << program_parameters[i];
  }
  OS << ");\n";
  // If this is a FTQC kernel, skip runtime optimization passes and submit.
  OS << "if (runtime_env == QrtType::FTQC) {\n";
  OS << "if (is_callable) {\n";
  // If this is the top-level kernel, during DTor we persit the bit value to
  // Buffer.
  OS << "quantum::persistBitstring(" << bufferNames[0] << ".results());\n";
  // Loop the function calls (at the top level only) if there are multiple shots
  // requested.
  OS << "for (size_t shotCount = 1; shotCount < quantum::get_shots(); "
        "++shotCount) {\n";
  OS << "operator()(" << program_parameters[0];
  for (int i = 1; i < program_parameters.size(); i++) {
    OS << ", " << program_parameters[i];
  }
  OS << ");\n";
  OS << "quantum::persistBitstring(" << bufferNames[0] << ".results());\n";
  OS << "}\n";
  // Use a special submit for FTQC to denote that this executable kernel has been completed.
  OS << "quantum::submit(nullptr);\n";
  OS << "}\n";
  OS << "return;\n";
  OS << "}\n";

  OS << "xacc::internal_compiler::execute_pass_manager();\n";
  OS << "if (optimize_only) {\n";
  OS << "return;\n";
  OS << "}\n";

  OS << "if (is_callable) {\n";
  if (bufferNames.size() > 1) {
    OS << "xacc::AcceleratorBuffer * buffers[" << bufferNames.size() << "] = {";
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
  OS << "}\n";

  // close the quantum kernel subclass
  OS << "};\n";

  // Add a function with the kernel_name that takes
  // a parent CompositeInstruction as its first arg
  OS << "void " << kernel_name
     << "(std::shared_ptr<qcor::CompositeInstruction> parent, "
     << program_arg_types[0] << " " << program_parameters[0];
  for (int i = 1; i < program_arg_types.size(); i++) {
    OS << ", " << program_arg_types[i] << " " << program_parameters[i];
  }
  OS << ") {\n";
  OS << "class " << kernel_name << " __ker__temp__(parent, "
     << program_parameters[0];
  for (int i = 1; i < program_parameters.size(); i++) {
    OS << ", " << program_parameters[i];
  }
  OS << ");\n";
  OS << "}\n";

  // Declare the previous forward declaration
  OS << "void __internal_call_function_" << kernel_name << "("
     << program_arg_types[0] << " " << program_parameters[0];
  for (int i = 1; i < program_arg_types.size(); i++) {
    OS << ", " << program_arg_types[i] << " " << program_parameters[i];
  }
  OS << ") {\n";
  OS << "class " << kernel_name << " __ker__temp__(" << program_parameters[0];
  for (int i = 1; i < program_parameters.size(); i++) {
    OS << ", " << program_parameters[i];
  }
  OS << ");\n";
  OS << "}\n";

  if (add_het_map_ctor) {
    // Remove "&" from type string before getting the Python variables in the
    // HetMap. Note: HetMap can't store references.
    const auto remove_ref_arg_type =
        [](const std::string &org_arg_type) -> std::string {
      // We intentially only support a very limited set of pass-by-ref types
      // from the HetMap.
      // Only do: double& and int&
      if (org_arg_type == "double&") {
        return "double";
      }
      if (org_arg_type == "int&") {
        return "int";
      }
      // Keep the type string.
      return org_arg_type;
    };

    // Strategy: we unpack the args in the HetMap and call
    // the appropriate ctor overload.

    // For reference ctor params (e.g. double& and int&),
    // we create a local variable to copy the arg from the HetMap
    // before passing to the ctor.
    // We have a special machanism to handle *pass-by-reference*
    // in the Python side.
    // Non-reference types will just use inline `args.get<T>(key)` to unpack
    // the arguments.

    // List of resolved argument strings for ctor calls.
    std::vector<std::string> arg_ctor_list;
    // Code to copy *ref* type arguments from the HetMap.
    // This *must* be injected before the ctor call.
    std::stringstream ref_type_copy_decl_ss;
    int var_counter = 0;
    // Only handle non-qreg args
    for (int i = 1; i < program_parameters.size(); i++) {
      // If this is a *supported* ref types: double&, int&, etc.
      if (remove_ref_arg_type(program_arg_types[i]) != program_arg_types[i]) {
        // Generate a temp var
        const std::string new_var_name =
            "__temp_var__" + std::to_string(var_counter++);
        // Copy the var from HetMap to the temp var
        ref_type_copy_decl_ss << remove_ref_arg_type(program_arg_types[i])
                              << " " << new_var_name << " = "
                              << "args.get<"
                              << remove_ref_arg_type(program_arg_types[i])
                              << ">(\"" << program_parameters[i] << "\");\n";

        // We just pass this copied var to the ctor
        // where it expects a reference type.
        arg_ctor_list.emplace_back(new_var_name);
      } else if (program_arg_types[i].rfind("KernelSignature", 0) == 0) {
        // This is a KernelSignature argument.
        // The one in HetMap is the function pointer represented as a hex string.
        const std::string new_var_name =
            "__temp_kernel_ptr_var__" + std::to_string(var_counter++);
        // Retrieve the function pointer from the HetMap
        // ref_type_copy_decl_ss << "std::cout << args.getString(\""
        //                       << program_parameters[i] << "\").c_str() << std::endl;\n";
        ref_type_copy_decl_ss << "void* " << new_var_name << " = "
                              << "(void *) strtoull(args.getString(\""
                              << program_parameters[i] << "\").c_str(), nullptr, 16);\n";
        // ref_type_copy_decl_ss << "std::cout << " << new_var_name << " << std::endl;\n";
        // Construct the KernelSignature
        const std::string kernel_signature_var_name =
            "__temp_kernel_signature_var__" + std::to_string(var_counter++);
        ref_type_copy_decl_ss << program_arg_types[i] << " "
                              << kernel_signature_var_name << "("
                              << new_var_name << ");\n";
        arg_ctor_list.emplace_back(kernel_signature_var_name);
      } else if (program_arg_types[i].rfind("std::vector<KernelSignature<",
                                            0) == 0) {
        // This is a list of KernelSignatures argument.
        // The one in HetMap is the vector of function pointers represented as a
        // hex string.
        const std::string new_var_name =
            "__temp_kernel_ptr_vector_var__" + std::to_string(var_counter++);
        // Retrieve the list of function pointer from the HetMap
        ref_type_copy_decl_ss << "auto " << new_var_name
                              << " = args.get<std::vector<std::string>>(\""
                              << program_parameters[i] << "\");\n";

        const std::string list_kernel_signature_var_name =
            "__temp_kernel_signature_var__" + std::to_string(var_counter++);
        // Declare the vector of kernel signatures
        ref_type_copy_decl_ss << program_arg_types[i] << " "
                              << list_kernel_signature_var_name << ";\n";
        // Construct the list from function pointers
        ref_type_copy_decl_ss << "std::vector<void*> temp_fn_ptrs(" << new_var_name << ".size());\n";
        ref_type_copy_decl_ss << "int fn_idx = 0;\n";
        ref_type_copy_decl_ss << "for (const auto& ptr_str: " << new_var_name << ") {\n";
        ref_type_copy_decl_ss << "temp_fn_ptrs[fn_idx] = (void *) strtoull(ptr_str.c_str(), nullptr, 16);\n";
        ref_type_copy_decl_ss <<  list_kernel_signature_var_name << ".emplace_back(temp_fn_ptrs[fn_idx++]);\n";
        ref_type_copy_decl_ss << "}\n";
        arg_ctor_list.emplace_back(list_kernel_signature_var_name);
      } else {
        // Otherwise, just unpack the arg inline in the ctor call.
        std::stringstream ss;
        ss << "args.get<" << program_arg_types[i] << ">(\""
           << program_parameters[i] << "\")";
        arg_ctor_list.emplace_back(ss.str());
      }
    }

    // Add the HeterogeneousMap args function overload
    OS << "void " << kernel_name
       << "__with_hetmap_args(HeterogeneousMap& args) {\n";
    // First, inject any copying statements required to unpack *ref* types.
    OS << ref_type_copy_decl_ss.str();
    // CTor call
    OS << "class " << kernel_name << " __ker__temp__(";
    // First arg: qreg
    OS << "args.get<" << program_arg_types[0] << ">(\"" << program_parameters[0]
       << "\")";
    // The rest: either inline unpacking or temp var names (ref type)
    for (const auto &arg_str : arg_ctor_list) {
      OS << ", " << arg_str;
    }
    OS << ");\n";
    OS << "}\n";

    OS << "void " << kernel_name
       << "__with_parent_and_hetmap_args(std::shared_ptr<CompositeInstruction> "
          "parent, "
          "HeterogeneousMap& args) {\n";
    OS << ref_type_copy_decl_ss.str();
    // CTor call with parent kernel
    OS << "class " << kernel_name << " __ker__temp__(parent, ";
    // Second arg: qreg
    OS << "args.get<" << program_arg_types[0] << ">(\"" << program_parameters[0]
       << "\")";
    // The rest: either inline unpacking or temp var names (ref type)
    for (const auto &arg_str : arg_ctor_list) {
      OS << ", " << arg_str;
    }
    OS << ");\n";
    // The rest: either inline unpacking or temp var names (ref type)
    OS << "}\n";
  }
  auto s = OS.str();
  qcor::info("[qcor syntax-handler] Rewriting " + kernel_name + " to\n\n" + s);
}

void QCORSyntaxHandler::AddToPredefines(llvm::raw_string_ostream &OS) {
  if (__internal__developer__flags__::add_predefines) {
    OS << "#include \"qcor.hpp\"\n";
    OS << "using namespace qcor;\n";
    OS << "using namespace xacc::internal_compiler;\n";
#ifdef _QCOR_MUTEX
    OS << "#include <mutex>\n";
#endif
  }
}

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

}  // namespace qcor

static SyntaxHandlerRegistry::Add<qcor::QCORSyntaxHandler> X(
    "qcor", "qcor quantum kernel syntax handler");

static FrontendPluginRegistry::Add<qcor::QCORArgs> XX("qcor-args", "");
