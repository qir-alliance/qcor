#pragma once
#include "Quantum/QuantumOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace qcor {
// Compile-time inlining of modifiled-block (if possible)
struct ModifierBlockInlinerPass
    : public PassWrapper<ModifierBlockInlinerPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() final;
  ModifierBlockInlinerPass() {}
private:
  void handlePowU();
  void handleCtrlU();
  void handleAdjU();
  void applyControlledQuantumOp(mlir::quantum::ValueSemanticsInstOp &qvsOp,
                                mlir::Value control_bit,
                                mlir::OpBuilder &rewriter);
};
} // namespace qcor