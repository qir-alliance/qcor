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
                               bool add_entry_point) {
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
  auto optPipeline = mlir::makeOptimizingTransformer(3, 0, nullptr);
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

/// Trait that defines how a given type is passed to the JIT code. This
/// defaults to passing the address but can be specialized.
template <typename T>
struct Argument {
  static void pack(SmallVectorImpl<void *> &args, T &val) {
    args.push_back(&val);
  }
};

/// Tag to wrap an output parameter when invoking a jitted function.
template <typename T>
struct Result {
  Result(T &result) : value(result) {}
  T &value;
};

/// Helper function to wrap an output operand when using
/// ExecutionEngine::invoke.
template <typename T>
static Result<T> result(T &t) {
  return Result<T>(t);
}

// Specialization for output parameter: their address is forwarded directly to
// the native code.
template <typename T>
struct Argument<Result<T>> {
  static void pack(SmallVectorImpl<void *> &args, Result<T> &result) {
    args.push_back(&result.value);
  }
};

/// Invokes the function with the given name passing it the list of arguments
/// by value. Function result can be obtain through output parameter using the
/// `Result` wrapper defined above. For example:
///
///     func @foo(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface }
///
/// can be invoked:
///
///     int32_t result = 0;
///     llvm::Error error = jit->invoke("foo", 42,
///                                     result(result));
template <typename... Args>
llvm::Error invoke(std::unique_ptr<mlir::ExecutionEngine> engine,
                   StringRef funcName, Args... args) {
  llvm::SmallVector<void *> argsArray;
  // Pack every arguments in an array of pointers. Delegate the packing to a
  // trait so that it can be overridden per argument type.
  // TODO: replace with a fold expression when migrating to C++17.
  int dummy[] = {0, ((void)Argument<Args>::pack(argsArray, args), 0)...};
  (void)dummy;
  return engine->invoke(funcName, argsArray);
}

void execute(const std::string &src_language_type, const std::string &src,
             const std::string &kernel_name) {
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

  // Create the PassManager for lowering to LLVM MLIR and run it
  mlir::PassManager pm(&context);
  pm.addPass(
      std::make_unique<qcor::QuantumToLLVMLoweringPass>(unique_function_names));
  auto module_op = (*module).getOperation();
  if (mlir::failed(pm.run(module_op))) {
    std::cout << "Pass Manager Failed\n";
    return;
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
    return;
  }

  QJIT jit;
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

  jit.invoke("main", cstrs.size(), cstrs.data());

  return;
}

}  // namespace qcor