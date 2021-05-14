
#pragma once

#include "Quantum/QuantumOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace qcor {

struct QuantumToLLVMLoweringPass
    : public PassWrapper<QuantumToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override;
  void runOnOperation() final;

 private:
  mlir::PassManager& pm;
  std::vector<std::string>& function_names;

 public:
  QuantumToLLVMLoweringPass(mlir::PassManager& p,
                            std::vector<std::string>& f_names)
      : pm(p), function_names(f_names) {}
};

// Helper func.
mlir::Type get_quantum_type(std::string type, mlir::MLIRContext* context);

// std::unique_ptr<mlir::Pass> createQuantumOptPass();
}  // namespace qcor