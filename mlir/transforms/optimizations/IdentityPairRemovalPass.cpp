/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "IdentityPairRemovalPass.hpp"
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
void SingleQubitIdentityPairRemovalPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}

void SingleQubitIdentityPairRemovalPass::runOnOperation() {
  // Walk the operations within the function.
  std::vector<mlir::quantum::ValueSemanticsInstOp> deadOps;
  getOperation().walk([&](mlir::quantum::ValueSemanticsInstOp op) {
    if (std::find(deadOps.begin(), deadOps.end(), op) != deadOps.end()) {
      // Skip this op since it was merged (forward search)
      return;
    }

    auto inst_name = op.name();
    auto return_value = *op.result().begin();
    if (return_value.hasOneUse()) {
      // get that one user
      auto user = *return_value.user_begin();
      // cast to a inst op
      if (auto next_inst =
              dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(user)) {
        // check that it is one of our known id pairs
        if (should_remove(next_inst.name().str(), inst_name.str())) {
          // need to get users of next_inst and point them to use
          // op.getOperands
          (*next_inst.result_begin()).replaceAllUsesWith(op.getOperand(0));
          deadOps.emplace_back(op);
          deadOps.emplace_back(next_inst);
        }
      }
    }
  });

  for (auto &op : deadOps) {
    op->dropAllUses();
    op.erase();
  }
}

bool SingleQubitIdentityPairRemovalPass::should_remove(
    std::string name1, std::string name2) const {
  if (search_gates.count(name1)) {
    return search_gates.at(name1) == name2;
  }
  return false;
}
void CNOTIdentityPairRemovalPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}
void CNOTIdentityPairRemovalPass::runOnOperation() {
  // Walk the operations within the function.
  std::vector<mlir::quantum::ValueSemanticsInstOp> deadOps;
  getOperation().walk([&](mlir::quantum::ValueSemanticsInstOp op) {
    if (std::find(deadOps.begin(), deadOps.end(), op) != deadOps.end()) {
      // Skip this op since it was merged (forward search)
      return;
    }

    auto inst_name = op.name();

    if (inst_name != "cnot" && inst_name != "cx") {
      return;
    }

    // Get the src ret qubit and the tgt ret qubit
    auto src_return_val = op.result().front();
    auto tgt_return_val = op.result().back();

    // Make sure they are used
    if (src_return_val.hasOneUse() && tgt_return_val.hasOneUse()) {

      // get the users of these values
      auto src_user = *src_return_val.user_begin();
      auto tgt_user = *tgt_return_val.user_begin();

      // Cast them to InstOps
      auto next_inst =
          dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(src_user);
      auto tmp_tgt =
          dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(tgt_user);
      if (!next_inst || !tmp_tgt) {
        // not inst ops
        return;
      }

      // We want the case where src_user and tgt_user are the same
      if (next_inst.getOperation() != tmp_tgt.getOperation()) {
        return;
      }

      // Need src_return_val to map to next_inst operand 0,
      // and tgt_return_val to map to next_inst operand 1.
      // if not drop out
      if (next_inst.getOperand(0) != src_return_val &&
          next_inst.getOperand(1) != tgt_return_val) {
        return;
      }

      // Next instruction must be a CNOT to merge
      if (next_inst.name() != "cnot" && next_inst.name() != "cx") {
        return;
      }

      // They are the same operation, a cnot
      // so we have cnot src, tgt | cnot src, tgt
      auto next_inst_result_0 = next_inst.result().front();
      auto next_inst_result_1 = next_inst.result().back();
      next_inst_result_0.replaceAllUsesWith(op.getOperand(0));
      next_inst_result_1.replaceAllUsesWith(op.getOperand(1));

      // Remove the identity pair
      deadOps.emplace_back(op);
      deadOps.emplace_back(src_user);
    }
  });

  for (auto &op : deadOps) {
    op->dropAllUses();
    op.erase();
  }
}

void DuplicateResetRemovalPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}

void DuplicateResetRemovalPass::runOnOperation() {
  // Walk the operations within the function.
  std::vector<mlir::quantum::ValueSemanticsInstOp> deadOps;
  getOperation().walk([&](mlir::quantum::ValueSemanticsInstOp op) {
    if (std::find(deadOps.begin(), deadOps.end(), op) != deadOps.end()) {
      // Skip this op since it was deleted (forward search)
      return;
    }
    auto inst_name = op.name();
    
    if (inst_name != "reset") {
      return;
    }

    auto return_value = *op.result().begin();
    if (return_value.hasOneUse()) {
      // get that one user
      auto user = *return_value.user_begin();
      // cast to a inst op
      if (auto next_inst =
              dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(user)) {
        if (next_inst.name().str() == "reset") {
          // Two resets in a row:
          // Chain the input -> output and mark this second reset to delete:
          (*next_inst.result_begin())
              .replaceAllUsesWith(next_inst.getOperand(0));
          deadOps.emplace_back(next_inst);
        }
      }
    }
  });

  for (auto &op : deadOps) {
    op->dropAllUses();
    op.erase();
  }
}
} // namespace qcor
