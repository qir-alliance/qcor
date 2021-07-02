#include "RemoveUnusedQIRCallsPass.hpp"
#include "Quantum/QuantumOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include <iostream>

namespace qcor {
void RemoveUnusedQIRCallsPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}
void RemoveUnusedQIRCallsPass::runOnOperation() {
  std::vector<Operation *> deadOps;
  getOperation().walk([&](mlir::quantum::ExtractQubitOp op) {
    // Extracted qubit has no use
    if (op.qbit().use_empty()) {
      deadOps.emplace_back(op.getOperation());
    }
  });

  // Remove any constant ops that are not being used. 
  getOperation().walk([&](mlir::ConstantOp op) {
    // Extracted qubit has no use
    if (op.getResult().use_empty()) {
      deadOps.emplace_back(op.getOperation());
    }
  });

  // Need to remove these extract calls to realize qalloc removal below.
  for (auto &op : deadOps) {
    op->dropAllUses();
    op->erase();
  }
  deadOps.clear();

  getOperation().walk([&](mlir::quantum::QallocOp op) {
    // QallocOp returns a qubit array, if it
    // only has one user, then that user must be
    // the dealloc op, so we can remove both of them
    auto qubits = op.qubits();
    auto next_op = (*qubits.user_begin());
    if (dyn_cast_or_null<mlir::quantum::DeallocOp>(next_op) &&
        qubits.hasOneUse()) {
      deadOps.emplace_back(op.getOperation());
      deadOps.emplace_back(next_op);
    }
  });
  for (auto &op : deadOps) {
    op->dropAllUses();
    op->erase();
  }
}
} // namespace qcor