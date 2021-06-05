
#include "token_collector_helper.hpp"
#include <sstream>

namespace __internal__ {
namespace qcor {

std::string construct_kernel_subtype(
    std::string new_src,
    const std::string kernel_name,
    const std::vector<std::string> &program_arg_types,
    const std::vector<std::string> &program_parameters,
    std::vector<std::string> bufferNames) {
  std::stringstream OS;

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
  OS << "void operator()(" << program_arg_types[0] << " "
     << program_parameters[0];
  for (int i = 1; i < program_arg_types.size(); i++) {
    OS << ", " << program_arg_types[i] << " " << program_parameters[i];
  }
  OS << ") {\n";
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

  return OS.str();
}
}  // namespace qcor
}  // namespace __internal__
