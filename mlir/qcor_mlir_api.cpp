#include "qcor_mlir_api.hpp"

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
#include "qcor_config.hpp"
#include "qcor_jit.hpp"
#include "quantum_to_llvm.hpp"
#include "tools/ast_printer.hpp"

namespace qcor {

const std::string mlir_compile(const std::string &src_language_type,
                               const std::string &src,
                               const std::string &kernel_name,
                               const OutputType &output_type,
                               bool add_entry_point, int opt_level) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();

  mlir::MLIRContext context;
  context.loadDialect<mlir::quantum::QuantumDialect, mlir::StandardOpsDialect,
                      mlir::scf::SCFDialect, mlir::AffineDialect>();

  std::vector<std::string> unique_function_names;

  std::shared_ptr<QuantumMLIRGenerator> mlir_generator;
  if (src_language_type == "openqasm") {
    mlir_generator = std::make_shared<OpenQasmMLIRGenerator>(context);
  } else if (src_language_type == "qasm3") {
    mlir_generator = std::make_shared<OpenQasmV3MLIRGenerator>(context);
  } else {
    std::cout << "No other mlir generators yet.\n";
    exit(1);
  }

  mlir_generator->initialize_mlirgen(add_entry_point, kernel_name);
  mlir_generator->mlirgen(src);
  mlir_generator->finalize_mlirgen();
  unique_function_names = mlir_generator->seen_function_names();
  auto module = mlir_generator->get_module();

  // std::cout << "MLIR + Quantum Dialect:\n";
  if (output_type == OutputType::MLIR) {
    std::string s;
    llvm::raw_string_ostream os(s);
    module->print(os);
    os.flush();
    return s;
  }

  // Create the PassManager for lowering to LLVM MLIR and run it
  mlir::PassManager pm(&context);
  pm.addPass(
      std::make_unique<qcor::QuantumToLLVMLoweringPass>(unique_function_names));
  auto module_op = (*module).getOperation();
  if (mlir::failed(pm.run(module_op))) {
    std::cout << "Pass Manager Failed\n";
    return "";
  }

  if (output_type == OutputType::LLVMMLIR) {
    std::string s;
    llvm::raw_string_ostream os(s);
    module->print(os);
    os.flush();
    return s;
  }

  // Now lower MLIR to LLVM IR
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);

  // Optimize the LLVM IR
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto optPipeline = mlir::makeOptimizingTransformer(opt_level, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return "";
  }

  if (output_type == OutputType::LLVMIR) {
    std::string s;
    llvm::raw_string_ostream os(s);
    llvmModule->print(os, nullptr, false, true);
    os.flush();
    return s;
  }

  exit(1);
  return "";
}

int execute(const std::string &src_language_type, const std::string &src,
            const std::string &kernel_name, int opt_level) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();

  mlir::MLIRContext context;
  context.loadDialect<mlir::quantum::QuantumDialect, mlir::StandardOpsDialect,
                      mlir::scf::SCFDialect, mlir::AffineDialect>();
  
  std::vector<std::string> unique_function_names;

  std::shared_ptr<QuantumMLIRGenerator> mlir_generator;
  if (src_language_type == "openqasm") {
    mlir_generator = std::make_shared<OpenQasmMLIRGenerator>(context);
  } else if (src_language_type == "qasm3") {
    mlir_generator = std::make_shared<OpenQasmV3MLIRGenerator>(context);
  } else {
    std::cout << "No other mlir generators yet.\n";
    exit(1);
  }

  mlir_generator->initialize_mlirgen(true, kernel_name);
  mlir_generator->mlirgen(src);
  mlir_generator->finalize_mlirgen();
  unique_function_names = mlir_generator->seen_function_names();
  auto module = mlir_generator->get_module();

DiagnosticEngine& engine = context.getDiagEngine();

/// Handle the reported diagnostic.
  // Return success to signal that the diagnostic has either been fully
  // processed, or failure if the diagnostic should be propagated to the
  // previous handlers.
  DiagnosticEngine::HandlerID id =
      engine.registerHandler([&](Diagnostic &diag) -> LogicalResult {
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
  mlir::PassManager pm(&context);
  pm.addPass(
      std::make_unique<qcor::QuantumToLLVMLoweringPass>(unique_function_names));
  auto module_op = (*module).getOperation();
  if (mlir::failed(pm.run(module_op))) {
    std::cout << "Pass Manager Failed\n";
    return 1;
  }

  // Now lower MLIR to LLVM IR
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);

  // Optimize the LLVM IR
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto optPipeline = mlir::makeOptimizingTransformer(opt_level, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return 1;
  }

  // Instantiate our JIT engine
  QJIT jit;

  // Compile the LLVM module, this is basically
  // just building up the LLVM JIT engine and
  // loading all seen function pointers
  jit.jit_compile(std::move(llvmModule),
                  std::vector<std::string>{std::string(QCOR_INSTALL_DIR) +
                                           std::string("/lib/libqir-qrt") +
                                           std::string(QCOR_LIB_SUFFIX)});

  std::vector<std::string> argv;
  std::vector<char *> cstrs;
  argv.insert(argv.begin(), "appExec");
  for (auto &s : argv) {
    cstrs.push_back(&s.front());
  }

  return jit.invoke_main(cstrs.size(), cstrs.data());
}

}  // namespace qcor