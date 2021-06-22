
#pragma once
#include "Quantum/QuantumOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace qcor {
// We make each optimization routine into its own Pass so that
// option such as `--print-ir-before-all` can be used to inspect
// each pass independently.
// TODO: make this a FunctionPass
struct SingleQubitGateMergingPass
    : public PassWrapper<SingleQubitGateMergingPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() final;
  SingleQubitGateMergingPass() {}
};
} // namespace qcor