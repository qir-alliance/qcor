/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#pragma once
#include <fstream>

#include "Quantum/QuantumDialect.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser.h"
#include "openqasm_mlir_generator.hpp"
#include "openqasmv3_mlir_generator.hpp"
#include "pass_manager.hpp"
#include "quantum_to_llvm.hpp"
#include "tools/ast_printer.hpp"
namespace qcor {
namespace util {
// Helper to run common MLIR tasks:
enum SourceLanguage { QASM2, QASM3 };
// Wrapper for MLIR generation results:
struct MlirGenerationResult {
  std::unique_ptr<mlir::MLIRContext> mlir_context;
  SourceLanguage source_language;
  std::vector<std::string> unique_function_names;
  std::unique_ptr<mlir::OwningModuleRef> module_ref;
  MlirGenerationResult(std::unique_ptr<mlir::MLIRContext> &&in_mlir_context,
                       SourceLanguage in_source_language,
                       const std::vector<std::string> &in_unique_function_names,
                       std::unique_ptr<mlir::OwningModuleRef> &&in_module_ref,
                       bool verbose_error = false)
      : mlir_context(std::move(in_mlir_context)),
        source_language(in_source_language),
        unique_function_names(in_unique_function_names),
        module_ref(std::move(in_module_ref)) {
    // Set the DiagnosticEngine handler
    DiagnosticEngine &engine = (*mlir_context).getDiagEngine();

    // Handle the reported diagnostic.
    // Return success to signal that the diagnostic has either been fully
    // processed, or failure if the diagnostic should be propagated to the
    // previous handlers.
    engine.registerHandler(
        [&, verbose_error](Diagnostic &diag) -> LogicalResult {
          if (verbose_error) {
            std::cout << "[qcor-mlir] Dumping Module after error.\n";
            (*module_ref)->dump();
          }
          std::string BOLD = "\033[1m";
          std::string RED = "\033[91m";
          std::string CLEAR = "\033[0m";

          for (auto &n : diag.getNotes()) {
            std::string s;
            llvm::raw_string_ostream os(s);
            n.print(os);
            os.flush();
            std::cout << BOLD << RED << "[qcor-mlir] Reported Error: " << s << "\n"
                      << CLEAR;
          }
          bool should_propagate_diagnostic = true;
          return failure(should_propagate_diagnostic);
        });
  }
  ~MlirGenerationResult() {}
};

std::pair<SourceLanguage, std::unique_ptr<mlir::OwningModuleRef>> loadMLIR(
    const std::string &qasm_src, const std::string &kernel_name,
    mlir::MLIRContext &context, std::vector<std::string> &function_names,
    bool addEntryPoint,
    std::map<std::string, std::string> extra_quantum_args = {}) {
  SourceLanguage src_language_type = SourceLanguage::QASM3;
  if (qasm_src.find("OPENQASM 2") != std::string::npos) {
    src_language_type = SourceLanguage::QASM2;
  }

  // FIXME Make this an extension point
  // llvm::StringRef(inputFilename).endswith(".qasm")
  // or llvm::StringRef(inputFilename).endswith(".quil")
  std::shared_ptr<qcor::QuantumMLIRGenerator> mlir_generator;
  if (src_language_type == SourceLanguage::QASM2) {
    mlir_generator = std::make_shared<qcor::OpenQasmMLIRGenerator>(context);
  } else if (src_language_type == SourceLanguage::QASM3) {
    mlir_generator = std::make_shared<qcor::OpenQasmV3MLIRGenerator>(context);
  } else {
    std::cout << "No other mlir generators yet.\n";
    exit(1);
  }

  mlir_generator->initialize_mlirgen(
      addEntryPoint, kernel_name,
      extra_quantum_args);  // FIXME HANDLE RELATIVE PATH
  mlir_generator->mlirgen(qasm_src);
  mlir_generator->finalize_mlirgen();
  function_names = mlir_generator->seen_function_names();
  return std::make_pair(src_language_type,
                        std::make_unique<mlir::OwningModuleRef>(
                            std::move(mlir_generator->get_module())));
}

MlirGenerationResult mlir_gen(
    const std::string &qasm_src, const std::string &kernel_name,
    bool add_entry_point,
    std::map<std::string, std::string> extra_quantum_args = {}) {
  auto context = std::make_unique<mlir::MLIRContext>();
  context->loadDialect<mlir::quantum::QuantumDialect, mlir::AffineDialect,
                       mlir::scf::SCFDialect, mlir::StandardOpsDialect>();

  std::vector<std::string> unique_function_names;
  auto [src_type, module] =
      loadMLIR(qasm_src, kernel_name, *context, unique_function_names,
               add_entry_point, extra_quantum_args);

  return MlirGenerationResult(std::move(context), src_type,
                              unique_function_names, std::move(module),
                              extra_quantum_args.count("verbose_error"));
}

MlirGenerationResult mlir_gen(
    const std::string &inputFilename, bool add_entry_point,
    std::string function_name = "",
    std::map<std::string, std::string> extra_quantum_args = {}) {
  llvm::StringRef ref(inputFilename);
  std::ifstream t(ref.str());
  std::string qasm_src((std::istreambuf_iterator<char>(t)),
                       std::istreambuf_iterator<char>());
  if (function_name.empty()) {
    function_name = llvm::sys::path::filename(inputFilename)
                        .split(StringRef("."))
                        .first.str();
  }
  return mlir_gen(qasm_src, function_name, add_entry_point, extra_quantum_args);
}
}  // namespace util
}  // namespace qcor
