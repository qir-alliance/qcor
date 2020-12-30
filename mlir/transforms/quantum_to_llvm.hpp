
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "quantum_dialect.hpp"

using namespace mlir;

namespace qcor {

struct QuantumToLLVMLoweringPass
    : public PassWrapper<QuantumToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() final;
 public:
  QuantumToLLVMLoweringPass() = default;
};

// std::unique_ptr<llvm::Module> lower_quantum_to_llvm(::mlir::ModuleOp module, mlir::MLIRContext& context);

}  // namespace qcor