
#include <fstream>
#pragma GCC diagnostic ignored "-Wpessimizing-move"

#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser.h"
#include "quantum_to_llvm.hpp"
#include "openqasm_mlir_generator.hpp"
#include "tools/ast_printer.hpp"

using namespace mlir;
using namespace staq;

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input openqasm file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
namespace {
enum Action {
  None,
  DumpMLIR,
  DumpLLVMIR
};
}
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")));
namespace {
enum InputType { QASM };
}
static cl::opt<enum InputType> inputType(
    "x", cl::init(QASM), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(QASM, "qasm",
                          "load the input file as a qasm source.")));

mlir::OwningModuleRef loadMLIR(mlir::MLIRContext &context) {
  llvm::StringRef ref(inputFilename);
  std::ifstream t(ref.str());
  std::string qasm_src((std::istreambuf_iterator<char>(t)),
                       std::istreambuf_iterator<char>());

  // FIXME Make this an extension point
  // llvm::StringRef(inputFilename).endswith(".qasm")
  // or llvm::StringRef(inputFilename).endswith(".quil")
  qcor::OpenQasmMLIRGenerator mlir_generator(context);
  mlir_generator.initialize_mlirgen();
  mlir_generator.mlirgen(qasm_src);
  mlir_generator.finalize_mlirgen();
  return mlir_generator.get_module();
}

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "qcor quantum assembly compiler\n");

  mlir::MLIRContext context;
  context.loadDialect<mlir::quantum::QuantumDialect, mlir::StandardOpsDialect,
                      mlir::vector::VectorDialect>();

  auto module = loadMLIR(context);

  // std::cout << "MLIR + Quantum Dialect:\n";
  if (emitAction == Action::DumpMLIR) {
    module->dump();
    return 0;
  }

  // Create the PassManager for lowering to LLVM MLIR and run it
  mlir::PassManager pm(&context);
  pm.addPass(std::make_unique<qcor::QuantumToLLVMLoweringPass>());
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
  auto optPipeline = mlir::makeOptimizingTransformer(3, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
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
  std::ofstream out_file(file_name+".ll");
  out_file << s;
  out_file.close();

  return 0;
}
