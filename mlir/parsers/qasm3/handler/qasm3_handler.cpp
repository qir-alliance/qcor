/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "qasm3_handler.hpp"

#include <iostream>
#include <regex>
#include <sstream>

#include "openqasmv3_mlir_generator.hpp"
#include "qasm3_handler_utils.hpp"
#include "quantum_to_llvm.hpp"
#include "pass_manager.hpp"

using namespace clang;

namespace qcor {

void Qasm3SyntaxHandler::GetReplacement(Preprocessor &PP, Declarator &D,
                                        CachedTokens &Toks,
                                        llvm::raw_string_ostream &OS) {
  // Get the function name
  auto kernel_name = D.getName().Identifier->getName().str();

  auto &diagnostics = PP.getDiagnostics();
  auto invalid_kernel_arg = diagnostics.getCustomDiagID(
      clang::DiagnosticsEngine::Fatal,
      "Invalid quantum kernel argument - we do not know how to map this type "
      "to a mlir::Type yet (%0 %1).");

  // Create the MLIRContext and load the dialects
  mlir::MLIRContext context;
  context
      .loadDialect<mlir::quantum::QuantumDialect, mlir::StandardOpsDialect>();

  // Create the mlir generator for qasm3
  OpenQasmV3MLIRGenerator mlir_generator(context);

  // Build up the function source code
  std::stringstream ss;
  for (auto &Tok : Toks) {
    ss << " ";
    ss << PP.getSpelling(Tok);
  }
  std::string src = ss.str();

  // Loop through function arguments and
  // build up associated mlir::Type arguments,
  // For vectors use Array *
  std::vector<mlir::Type> arg_types;
  std::vector<std::string> program_parameters, arg_type_strs, var_attributes;
  const DeclaratorChunk::FunctionTypeInfo &FTI = D.getFunctionTypeInfo();
  for (unsigned int ii = 0; ii < FTI.NumParams; ii++) {
    // Get parameters as a ParmVarDecl
    auto &paramInfo = FTI.Params[ii];
    auto &decl = paramInfo.Param;
    auto parm_var_decl = cast<ParmVarDecl>(decl);
    // Get the type pointer
    auto type = parm_var_decl->getType().getTypePtr();

    // Get VarName and Type as strings
    Token IdentToken, TypeToken, test;
    PP.getRawToken(paramInfo.IdentLoc, IdentToken);
    PP.getRawToken(decl->getBeginLoc(), TypeToken);

    auto var = PP.getSpelling(IdentToken);
    auto type_str = PP.getSpelling(TypeToken);

    // Convert type to a mlir type
    mlir::Type t = convertClangType(type, type_str, context);
    if (!t) {
      auto db = diagnostics.Report(invalid_kernel_arg);
      db.AddString(type->getCanonicalTypeInternal().getAsString());
      db.AddString(var);
    }

    arg_types.push_back(t);

    // Add them to the vectors
    program_parameters.push_back(var);
    arg_type_strs.push_back(type_str);

    if (type->getCanonicalTypeInternal().getAsString().find("vector<double>") !=
        std::string::npos) {
      var_attributes.push_back("double");
    } else {
      var_attributes.push_back("");
    }
  }

  // std::cout << "SRC:\n" << ss.str() << "\n";

  // Get the return type as an mlir type,
  // as well as a string
  std::string ret_type_str = "";
  mlir::Type return_type =
      convertReturnType(D.getDeclSpec(), ret_type_str, context);

  // Init the MLIRGen
  mlir_generator.initialize_mlirgen(kernel_name, arg_types, program_parameters,
                                    var_attributes, return_type);

  // Run the MLIRGen
  mlir_generator.mlirgen(src);

  // Finalize and get the Module
  mlir_generator.finalize_mlirgen();
  auto module = mlir_generator.get_module();
  // module->dump();

  // Lower the module to LLVM IR bit code file
  DiagnosticEngine &engine = context.getDiagEngine();

  // Handle the reported diagnostic.
  // Return success to signal that the diagnostic has either been fully
  // processed, or failure if the diagnostic should be propagated to the
  // previous handlers.
  engine.registerHandler([&](mlir::Diagnostic &diag) -> LogicalResult {
    std::cout << "Dumping Module after error.\n";
    module->dump();
    for (auto &n : diag.getNotes()) {
      std::string s;
      llvm::raw_string_ostream os(s);
      n.print(os);
      os.flush();
      std::cout << "DiagnosticEngine Note: " << s << "\n";
    }
    bool should_propagate_diagnostic = true;
    return failure(should_propagate_diagnostic);
  });

  // Create the PassManager for lowering to LLVM MLIR and run it
  std::vector<std::string> unique_f_names{kernel_name};
  mlir::PassManager pm(&context);
  applyPassManagerCLOptions(pm);
  qcor::configureOptimizationPasses(pm);
  pm.addPass(std::make_unique<qcor::QuantumToLLVMLoweringPass>(true, unique_f_names));
  auto module_op = (*module).getOperation();
  if (mlir::failed(pm.run(module_op))) {
    std::cout << "Pass Manager Failed\n";
  }
  // Now lower MLIR to LLVM IR
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);

  // Optimize the LLVM IR
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto optPipeline = mlir::makeOptimizingTransformer(3, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
  }

  // llvmModule->dump();
  // Write the LLVM IR to a bitcode file
  std::error_code ec;
  llvm::ToolOutputFile result(kernel_name + ".bc", ec, llvm::sys::fs::F_None);
  WriteBitcodeToFile(*llvmModule, result.os());
  result.keep();

  // Update the source code
  std::stringstream sss;
  sss << "extern \"C\" { " << ret_type_str << " __internal_mlir_" << kernel_name
      << "(" << arg_type_strs[0];
  for (int i = 1; i < arg_type_strs.size(); i++) {
    std::string type_name = arg_type_strs[i];
    if (arg_type_strs[i].find("qcor::qreg") != std::string::npos) {
      type_name = "Array*";
    } else if (arg_type_strs[i].find("std::vector") != std::string::npos) {
      type_name = "Array*";
    }
    sss << ", " << type_name;
  }
  sss << ");}\n";

  // Rewrite the function to call the internal function
  sss << getDeclText(PP, D).str() << "{\n";

  // Perform any argument translation
  // e.g. map qcor::qreg to Array* with q.raw_array();
  for (int i = 0; i < arg_type_strs.size(); i++) {
    if (arg_type_strs[i].find("qcor::qreg") != std::string::npos) {
      auto old_var_name = program_parameters[i];
      sss << "auto __tmp_internal_qreg_array_" << old_var_name << " = "
          << old_var_name << ".raw_array();\n";
      std::replace(program_parameters.begin(), program_parameters.end(),
                   old_var_name, "__tmp_internal_qreg_array_" + old_var_name);
    } else if (arg_type_strs[i].find("std::vector") != std::string::npos) {
      auto old_var_name = program_parameters[i];
      sss << "auto __tmp_internal_vector_array_" << old_var_name
          << " = qcor::qir::toArray(" << old_var_name << ");\n";
      std::replace(program_parameters.begin(), program_parameters.end(),
                   old_var_name, "__tmp_internal_vector_array_" + old_var_name);
    }
  }
  sss << "if (!initialized) initialize();\n";
  sss << "return __internal_mlir_" << kernel_name << "("
      << program_parameters[0];
  for (int i = 1; i < program_parameters.size(); i++) {
    sss << ", " << program_parameters[i];
  }
  sss << ");\n";
  sss << "}\n";

  // std::cout << "NEW CODE:\n" << sss.str() << "\n";
  OS << sss.str();
  return;
}

void Qasm3SyntaxHandler::AddToPredefines(llvm::raw_string_ostream &OS) {
  OS << "#include \"qir-qrt.hpp\"\n";
  OS << "#include \"qir-types-utils.hpp\"\n";
  OS << "using qcor::qubit;\n";
}
}  // namespace qcor

static SyntaxHandlerRegistry::Add<qcor::Qasm3SyntaxHandler> X(
    "qasm3", "qasm3 quantum kernel syntax handler");

// /usr/local/aideqc/llvm/bin/clang++ -std=c++17
// -fplugin=/home/cades/.xacc/clang-plugins/libqasm3-syntax-handler.so -c
// qasm3_test.cpp /usr/local/aideqc/llvm/bin/llc -filetype=obj test.bc
// /usr/local/aideqc/llvm/bin/clang++ test.o qasm3_test.o
// ./a.out