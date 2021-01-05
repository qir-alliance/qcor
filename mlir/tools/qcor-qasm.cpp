
#include <fstream>

#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "quantum_to_llvm.hpp"
#include "staq_parser.hpp"
#include "tools/ast_printer.hpp"

using namespace mlir;
using namespace staq;

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input openqasm file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "openqasm compiler\n");

  llvm::StringRef ref(inputFilename);
  std::ifstream t(ref.str());
  std::string qasm_src((std::istreambuf_iterator<char>(t)),
                       std::istreambuf_iterator<char>());

//   std::cout << "Original:\n" << qasm_src << "\n";
  // Parse the OpenQasm with Staq
  ast::ptr<ast::Program> prog;
  try {
    prog = parser::parse_string(qasm_src);
    // transformations::inline_ast(*prog);
    transformations::desugar(*prog);
  } catch (std::exception &e) {
    std::stringstream ss;
    std::cout << e.what() << "\n";
  }

  tools::print_tree(*prog, std::cout);

  std::cout << "After parsing:\n" << *prog << "\n";
  mlir::MLIRContext context;
  context.loadDialect<mlir::quantum::QuantumDialect, mlir::StandardOpsDialect,
                      mlir::vector::VectorDialect>();

  // Generate the MLIR using a Staq Visitor
  qasm_parser::StaqToMLIR visitor(context);
  visitor.visit(*prog);
  visitor.addReturn();
  auto module = visitor.module();

  std::cout << "MLIR + Quantum Dialect:\n";
  module->dump();

  // Create the PassManager for lowering to LLVM MLIR and run it
  mlir::PassManager pm(&context);
  pm.addPass(std::make_unique<qcor::QuantumToLLVMLoweringPass>());
  auto module_op = module.getOperation();
  if (mlir::failed(pm.run(module_op))) {
    std::cout << "Pass Manager Failed\n";
    return 1;
  }
//   std::cout << "Lowered to LLVM MLIR Dialect:\n";
//   module_op->dump();

  // Now lower MLIR to LLVM IR
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
//   std::cout << "Lowered to LLVM IR:\n";
//   llvmModule->dump();

  // Optimize the LLVM IR
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto optPipeline = mlir::makeOptimizingTransformer(3, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
//   std::cout << "Optimized LLVM IR:\n";
  llvmModule->dump();


  return 0;
}



// declare i64 @__quantum__qis__mz(i64* %q) 
// declare void @__quantum__qis__cnot(Qubit* %ctrl, Qubit* %tgt) 
// declare void @__quantum__qis__h(Qubit* %q) 
// declare Qubit* @__quantum__rt__array_get_element_ptr_1d(Qubit* %qreg, i64 %idx)
// declare Qubit* @__quantum__rt__qubit_allocate_array(i64 %size, ...) 

// define void @main() {
//   %1 = call Qubit* @__quantum__rt__qubit_allocate_array(i64 2)
//   %2 = call Qubit* @__quantum__rt__array_get_element_ptr_1d(Qubit* %1, i64 0)
//   call void @__quantum__qis__h(Qubit* %2)
//   %3 = call Qubit* @__quantum__rt__array_get_element_ptr_1d(Qubit* %1, i64 1)
//   tail void @__quantum__qis__cnot(Qubit* %2, Qubit* %3)
//   %4 = call i64 @__quantum__qis__mz(Qubit* %2)
//   %5 = call i64 @__quantum__qis__mz(Qubit* %3)
//   ret void
// }