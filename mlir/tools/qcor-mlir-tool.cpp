
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
cl::opt<bool> noEntryPoint("no-entrypoint",
                           cl::desc("Do not add main() to compiled output."));

cl::opt<bool> mlir_quantum_opt(
    "q-optimize",
    cl::desc("Turn on MLIR-level quantum instruction optimizations."));

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

mlir::OwningModuleRef loadMLIR(mlir::MLIRContext &context,
                               std::vector<std::string> &function_names) {
  llvm::StringRef ref(inputFilename);
  std::ifstream t(ref.str());
  std::string qasm_src((std::istreambuf_iterator<char>(t)),
                       std::istreambuf_iterator<char>());

  std::string src_language_type = "qasm3";
  if (qasm_src.find("OPENQASM 2") != std::string::npos) {
    src_language_type = "qasm2";
  }

  // FIXME Make this an extension point
  // llvm::StringRef(inputFilename).endswith(".qasm")
  // or llvm::StringRef(inputFilename).endswith(".quil")
  std::shared_ptr<QuantumMLIRGenerator> mlir_generator;
  if (src_language_type == "qasm2") {
    mlir_generator = std::make_shared<OpenQasmMLIRGenerator>(context);
  } else if (src_language_type == "qasm3") {
    mlir_generator = std::make_shared<OpenQasmV3MLIRGenerator>(context);
  } else {
    std::cout << "No other mlir generators yet.\n";
    exit(1);
  }
  auto function_name = llvm::sys::path::filename(inputFilename)
                           .split(StringRef("."))
                           .first.str();
  bool addEntryPoint = !noEntryPoint;
  mlir_generator->initialize_mlirgen(
      addEntryPoint, function_name);  // FIXME HANDLE RELATIVE PATH
  mlir_generator->mlirgen(qasm_src);
  mlir_generator->finalize_mlirgen();
  function_names = mlir_generator->seen_function_names();
  return mlir_generator->get_module();
}

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "qcor quantum assembly compiler\n");
  bool qoptimizations = mlir_quantum_opt;
  mlir::MLIRContext context;
  context.loadDialect<mlir::quantum::QuantumDialect, mlir::AffineDialect,
                      mlir::scf::SCFDialect, mlir::StandardOpsDialect>();

  std::vector<std::string> unique_function_names;
  auto module = loadMLIR(context, unique_function_names);

  DiagnosticEngine &engine = context.getDiagEngine();

  // Handle the reported diagnostic.
  // Return success to signal that the diagnostic has either been fully
  // processed, or failure if the diagnostic should be propagated to the
  // previous handlers.
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

  if (emitAction == Action::DumpMLIR) {
    module->dump();
    return 0;
  }

  // Create the PassManager for lowering to LLVM MLIR and run it
  mlir::PassManager pm(&context);
  applyPassManagerCLOptions(pm);
  pm.addPass(std::make_unique<qcor::QuantumToLLVMLoweringPass>(
      qoptimizations, unique_function_names));
  auto module_op = (*module).getOperation();
  if (mlir::failed(pm.run(module_op))) {
    std::cout << "Pass Manager Failed\n";
    return 1;
  }

  if (emitAction == Action::DumpMLIRLLVM) {
    module->dump();
    return 0;
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
  std::ofstream out_file(file_name + ".ll");
  out_file << s;
  out_file.close();

  return 0;
}
