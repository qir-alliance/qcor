#pragma once
#include "Quantum/QuantumOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace qcor {
// Merging 2 consecutive CPhase gates
// on the same control and target qubits.
struct CPhaseRotationMergingPass
    : public PassWrapper<CPhaseRotationMergingPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() final;
  CPhaseRotationMergingPass() {}

private:
  static constexpr double ZERO_ROTATION_TOLERANCE = 1e-9;
};
} // namespace qcor