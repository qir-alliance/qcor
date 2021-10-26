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
#include "qcor-mlir-helper.hpp"
#include "quantum_to_llvm.hpp"
#include "tools/ast_printer.hpp"

using namespace mlir;
using namespace staq;
using namespace qcor;

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input openqasm file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string> qpu(
    "qpu", cl::desc("The quantum coprocessor to compile to."));
static cl::opt<std::string> qrt(
    "qrt", cl::desc("The quantum execution mode: ftqc or nisq."));
static cl::opt<std::string> shots(
    "shots", cl::desc("The number of shots for nisq mode execution."));

cl::opt<bool> noEntryPoint("no-entrypoint",
                           cl::desc("Do not add main() to compiled output."));

cl::opt<bool> mlir_quantum_opt(
    "q-optimize",
    cl::desc("Turn on MLIR-level quantum instruction optimizations."));

cl::opt<std::string> mlir_specified_func_name(
    "internal-func-name", cl::desc("qcor provided function name"));

static cl::opt<bool> verbose_error(
    "verbose-error", cl::desc("Printout the full MLIR Module on error."));

static cl::opt<bool> print_final_submission(
    "print-final-submission", cl::desc("Print the XACC IR representation for submitted quantum code."));

static cl::opt<bool> OptLevelO0(
    "O0", cl::desc("Optimization level 0. Similar to clang -O0. "));

static cl::opt<bool> OptLevelO1(
    "O1", cl::desc("Optimization level 1. Similar to clang -O1. "));

static cl::opt<bool> OptLevelO2(
    "O2", cl::desc("Optimization level 2. Similar to clang -O2. "));

static cl::opt<bool> OptLevelO3(
    "O3", cl::desc("Optimization level 3. Similar to clang -O3. "));

namespace {
enum Action { None, DumpMLIR, DumpMLIRLLVM, DumpLLVMIR };
}
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR LLVM Dialect dump")));
namespace {
enum InputType { QASM };
}
static cl::opt<enum InputType> inputType(
    "x", cl::init(QASM), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(QASM, "qasm",
                          "load the input file as a qasm source.")));

static cl::opt<bool> mlir_debug_dialect_conversion(
    "debug-dialect-conversion",
    cl::desc("Debug the execution of the dialect conversion framework. Similar "
             "to '-debug-only=dialect-conversion'."));

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "qcor quantum assembly compiler\n");

  if (mlir_debug_dialect_conversion) {
    llvm::DebugFlag = true;
    llvm::setCurrentDebugType("dialect-conversion");
  }

  // If any *clang* optimization is requested, turn on quantum optimization as
  // well.
  bool qoptimizations =
      mlir_quantum_opt || OptLevelO1 || OptLevelO2 || OptLevelO3;
  std::string input_func_name = "";
  if (!mlir_specified_func_name.empty()) {
    input_func_name = mlir_specified_func_name;
  }

  // Check for extra quantum compiler flags.
  std::map<std::string, std::string> extra_args;
  if (!qpu.empty()) {
    extra_args.insert({"qpu", qpu});
  }
  if (!qrt.empty()) {
    extra_args.insert({"qrt", qrt});
  }
  if (!shots.empty()) {
    extra_args.insert({"shots", shots});
  }

  if (verbose_error) {
    extra_args.insert({"verbose_error", ""});
  }

  if (print_final_submission) {
    extra_args.insert({"print_final_submission", ""});
  }

  auto mlir_gen_result = qcor::util::mlir_gen(inputFilename, !noEntryPoint,
                                              input_func_name, extra_args);
  mlir::OwningModuleRef &module = *(mlir_gen_result.module_ref);
  mlir::MLIRContext &context = *(mlir_gen_result.mlir_context);
  std::vector<std::string> &unique_function_names =
      mlir_gen_result.unique_function_names;

  // Create the PassManager for lowering to LLVM MLIR and run it
  mlir::PassManager pm(&context);
  applyPassManagerCLOptions(pm);

  std::string BOLD = "\033[1m";
  std::string RED = "\033[91m";
  std::string CLEAR = "\033[0m";

  if (qoptimizations) {
    // Add optimization passes
    qcor::configureOptimizationPasses(pm);
  }

  if (emitAction == Action::DumpMLIR) {
    if (qoptimizations) {
      auto module_op = (*module).getOperation();
      if (mlir::failed(pm.run(module_op))) {
        std::cout << BOLD << RED
                  << "[qcor-mlir-tool] Language-to-MLIR lowering failed.\n"
                  << CLEAR;
        return 1;
      }
    }
    module->dump();
    return 0;
  }

  // Lower MLIR to LLVM
  pm.addPass(std::make_unique<qcor::ModifierRegionRewritePass>());
  pm.addPass(std::make_unique<qcor::QuantumToLLVMLoweringPass>(
      qoptimizations, unique_function_names));

  auto module_op = (*module).getOperation();
  if (mlir::failed(pm.run(module_op))) {
    std::cout << BOLD << RED
              << "[qcor-mlir-tool] MLIR-to-LLVM_MLIR lowering failed.\n"
              << CLEAR;
    return 1;
  }

  if (emitAction == Action::DumpMLIRLLVM) {
    module->dump();
    return 0;
  }

  // Now lower MLIR to LLVM IR
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    std::cout << BOLD << RED
              << "[qcor-mlir-tool] MLIR_LLVM-to-LLVM_IR lowering failed.\n"
              << CLEAR;
    return -1;
  }
  // Optimize the LLVM IR
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  const unsigned optLevel =
      OptLevelO3 ? 3 : (OptLevelO2 ? 2 : (OptLevelO1 ? 1 : 0));
  // std::cout << "Opt level: " << optLevel << "\n";
  auto optPipeline = mlir::makeOptimizingTransformer(optLevel, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << BOLD << RED << "Failed to optimize LLVM IR " << err << "\n"
                 << CLEAR;
    return -1;
  }

  if (emitAction == Action::DumpLLVMIR) {
    llvmModule->dump();
    return 0;
  }

  // llvmModule->dump();
  // Write to an ll file
  std::string s;
  llvm::raw_string_ostream os(s);
  llvmModule->print(os, nullptr, false, true);
  os.flush();
  auto file_name =
      llvm::StringRef(inputFilename).split(StringRef(".")).first.str();
  std::ofstream out_file(file_name + ".ll");
  out_file << s;
  out_file.close();

  return 0;
}
