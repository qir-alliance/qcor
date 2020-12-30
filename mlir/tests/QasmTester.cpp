#pragma GCC diagnostic ignored "-Wsuggest-override"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wpessimizing-move"

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "optimization/simplify.hpp"
#include "quantum_dialect.hpp"
#include "staq_parser.hpp"
#include "transformations/desugar.hpp"
#include "transformations/inline.hpp"

#include "quantum_to_llvm.hpp"

using namespace mlir;
using namespace staq;

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  std::string lineText = R"#(OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
CX q[1], q[0];
U(1.1,2.2,3.3) q[1];
rx(2.333) q[0];
creg c[2];
measure q -> c;
)#";

  std::cout << "Original:\n" << lineText << "\n";
  // Parse the OpenQasm with Staq
  ast::ptr<ast::Program> prog;
  try {
    prog = parser::parse_string(lineText);
    transformations::desugar(*prog);
    // transformations::synthesize_oracles(*prog);
  } catch (std::exception &e) {
    std::stringstream ss;
    std::cout << e.what() << "\n";
  }

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::quantum::QuantumDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  context.getOrLoadDialect<mlir::vector::VectorDialect>();

  // Generate the MLIR using a Staq Visitor
  qasm_parser::StaqToMLIR visitor(context);
  visitor.visit(*prog);
  visitor.addReturn();

  std::cout << "MLIR + Quantum Dialect:\n";
  visitor.module()->dump();

  // Create the PassManager for lowering to LLVM MLIR and run it
  mlir::PassManager pm(&context);
  pm.addPass(std::make_unique<qcor::QuantumToLLVMLoweringPass>());
  auto module = visitor.module();
  auto module_op = module.getOperation();
  auto result = pm.run(module_op);
  std::cout << "Lowered to LLVM MLIR Dialect:\n";
  module_op->dump();

  // Now lower MLIR to LLVM IR
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  std::cout << "Lowered to LLVM IR:\n";
  llvmModule->dump();

  // Optimize the LLVM IR
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  bool enableOpt = true;
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  std::cout << "Optimized LLVM IR:\n";
  llvmModule->dump();
  return 0;
}
