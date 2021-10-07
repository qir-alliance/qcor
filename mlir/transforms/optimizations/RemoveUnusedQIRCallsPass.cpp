/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
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

  // Strategy for alias construction clean-up:
  // In general, we must trim the alias construction from bottom up:
  // i.e. array concatenation (||) must be check (for no use) first then remove
  // if the resulting array is unused. This can then free up the two left and
  // right operands for potential clean-up (by checking for no use).

  // Remove any qubit array concat/create that has no use:
  // (from qubit array aliasing)
  getOperation().walk([&](mlir::quantum::ArrayConcatOp op) {
    // ArrayConcatOp has no use
    if (op.concat_array().use_empty()) {
      deadOps.emplace_back(op.getOperation());
    }
  });
  for (auto &op : deadOps) {
    op->dropAllUses();
    op->erase();
  }
  deadOps.clear();

  getOperation().walk([&](mlir::quantum::ArraySliceOp op) {
    // ArraySliceOp has no use
    if (op.array_slice().use_empty()) {
      deadOps.emplace_back(op.getOperation());
    }
  });
  for (auto &op : deadOps) {
    op->dropAllUses();
    op->erase();
  }
  deadOps.clear();

  // Remove alias array construction that has no use:
  // i.e. createQubitArray then qassign but no other uses:
  getOperation().walk([&](mlir::quantum::QaliasArrayAllocOp op) {
    // QaliasArrayAllocOp has no use except qassign 
    // i.e. we construct the alias array but not using it
    bool allUsesAreQassign = true;

    for (auto user_iter = op.qubits().user_begin();
         user_iter != op.qubits().user_end(); ++user_iter) {
      auto user = *user_iter;
      // cast to a AssignQubitOp; break if not a AssignQubitOp
      if (!dyn_cast_or_null<mlir::quantum::AssignQubitOp>(user)) {
        allUsesAreQassign = false;
        break;
      }
    }

    if (allUsesAreQassign) {
      deadOps.emplace_back(op.getOperation());
      for (auto user_iter = op.qubits().user_begin();
           user_iter != op.qubits().user_end(); ++user_iter) {
        auto user = *user_iter;
        auto assignOp = dyn_cast_or_null<mlir::quantum::AssignQubitOp>(user);
        deadOps.emplace_back(assignOp.getOperation());
      }
    }
  });
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
  deadOps.clear();

  // Run another round of ConstantOp trimming
  // (potentially realizable after the above optimizations)
  getOperation().walk([&](mlir::ConstantOp op) {
    // ConstantOp has no use
    if (op.getResult().use_empty()) {
      deadOps.emplace_back(op.getOperation());
    }
  });
  for (auto &op : deadOps) {
    op->dropAllUses();
    op->erase();
  }
  deadOps.clear();
}
} // namespace qcor