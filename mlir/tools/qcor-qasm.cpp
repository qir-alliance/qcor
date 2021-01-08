
#include <fstream>
#pragma GCC diagnostic ignored "-Wpessimizing-move"

#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Parser.h"
#include "quantum_to_llvm.hpp"
#include "staq_parser.hpp"
#include "tools/ast_printer.hpp"
#include "mlir/IR/AsmState.h"

using namespace mlir;
using namespace staq;

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input openqasm file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
namespace {
enum InputType { QASM, MLIR };
}
static cl::opt<enum InputType> inputType(
    "x", cl::init(QASM), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(QASM, "qasm",
                          "load the input file as a qasm source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
ast::ptr<ast::Program> parseInputFile(llvm::StringRef filename) {
  llvm::StringRef ref(inputFilename);
  std::ifstream t(ref.str());
  std::string qasm_src((std::istreambuf_iterator<char>(t)),
                       std::istreambuf_iterator<char>());
  ast::ptr<ast::Program> prog;
  try {
    prog = parser::parse_string(qasm_src);
    // transformations::inline_ast(*prog);
    transformations::desugar(*prog);
  } catch (std::exception &e) {
    std::stringstream ss;
    std::cout << e.what() << "\n";
  }

  return prog;
}

mlir::OwningModuleRef loadMLIR(mlir::MLIRContext &context) {
  // Handle '.toy' input to the compiler.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).endswith(".mlir")) {
    auto moduleAST = parseInputFile(inputFilename);

    qasm_parser::StaqToMLIR visitor(context);
    visitor.visit(*moduleAST);
    visitor.addReturn();
    return mlir::OwningModuleRef(
        mlir::OwningOpRef<mlir::ModuleOp>(visitor.module()));
  }

  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  return mlir::parseSourceFile(sourceMgr, &context);
}

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "openqasm compiler\n");

  mlir::MLIRContext context;
  context.loadDialect<mlir::quantum::QuantumDialect, mlir::StandardOpsDialect,
                      mlir::vector::VectorDialect>();

  auto module = loadMLIR(context);

  // std::cout << "MLIR + Quantum Dialect:\n";
  // module->dump();

  // Create the PassManager for lowering to LLVM MLIR and run it
  mlir::PassManager pm(&context);
  pm.addPass(std::make_unique<qcor::QuantumToLLVMLoweringPass>());
  auto module_op = (*module).getOperation();
  if (mlir::failed(pm.run(module_op))) {
    std::cout << "Pass Manager Failed\n";
    return 1;
  }
  // std::cout << "Lowered to LLVM MLIR Dialect:\n";
  // module_op->dump();

  // Now lower MLIR to LLVM IR
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
  // std::cout << "\nLowered to LLVM IR:\n";

  // Optimize the LLVM IR
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto optPipeline = mlir::makeOptimizingTransformer(3, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  //   std::cout << "Optimized LLVM IR:\n";
  // llvmModule->dump();

  std::string s;
  llvm::raw_string_ostream os(s);
  llvmModule->print(os, nullptr, false, true);
  os.flush();

  // std::cout << "P HERE\n" << s << "\n";

  std::string file_name = "bell.ll";
  std::ofstream out_file(file_name);
  out_file << s;
  out_file.close();

  return 0;
}
