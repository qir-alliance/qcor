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
#include "CphaseRotationMergingPass.hpp"
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
#include <optional>

namespace qcor {
void CPhaseRotationMergingPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}

void CPhaseRotationMergingPass::runOnOperation() {
  // Walk the operations within the function.
  std::vector<mlir::quantum::ValueSemanticsInstOp> deadOps;
  getOperation().walk([&](mlir::quantum::ValueSemanticsInstOp op) {
    if (std::find(deadOps.begin(), deadOps.end(), op) != deadOps.end()) {
      // Skip this op since it was merged (forward search)
      return;
    }
    mlir::OpBuilder rewriter(op);
    auto inst_name = op.name();
    if (inst_name != "cphase") {
      return;
    }
    assert(op.getOperands().size() == 3);
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

      // Ctrl -> ctrl; target -> target connections
      if (next_inst.getOperand(0) != src_return_val &&
          next_inst.getOperand(1) != tgt_return_val) {
        return;
      }

      if (next_inst.name() != "cphase") {
        return;
      }

      // They are the same operation, a cphase
      // so we have cphase src, tgt | cphase src, tgt
      // Combine the angles:
      const auto tryGetConstAngle =
          [](mlir::Value theta_var) -> std::optional<double> {
        if (!theta_var.getType().isa<mlir::FloatType>()) {
          return std::nullopt;
        }
        // Find the defining op:
        auto def_op = theta_var.getDefiningOp();
        if (def_op) {
          // Try cast:
          if (auto const_def_op =
                  dyn_cast_or_null<mlir::ConstantFloatOp>(def_op)) {
            llvm::APFloat theta_var_const_cal = const_def_op.getValue();
            return theta_var_const_cal.convertToDouble();
          }
        }
        return std::nullopt;
      };

      mlir::Value first_angle = op.getOperand(2);
      mlir::Value second_angle = next_inst.getOperand(2);
      rewriter.setInsertionPointAfter(next_inst);

      const auto first_angle_const = tryGetConstAngle(first_angle);
      const auto second_angle_const = tryGetConstAngle(second_angle);

      // Create a new instruction:
      // Return type: qubit; qubit
      std::vector<mlir::Type> ret_types{op.getOperand(0).getType(),
                                        op.getOperand(1).getType()};
      const std::string result_inst_name = "cphase";
      if (first_angle_const.has_value() && second_angle_const.has_value()) {
        // both angles are constant: pre-compute the total angle:
        const double totalAngle =
            first_angle_const.value() + second_angle_const.value();
        if (std::abs(totalAngle) > ZERO_ROTATION_TOLERANCE) {
          mlir::Value totalAngleVal = rewriter.create<mlir::ConstantOp>(
              op.getLoc(),
              mlir::FloatAttr::get(rewriter.getF64Type(), totalAngle));
          auto new_inst = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
              op.getLoc(), llvm::makeArrayRef(ret_types), result_inst_name,
              llvm::makeArrayRef({op.getOperand(0), op.getOperand(1)}),
              llvm::makeArrayRef({totalAngleVal}));
          // Input -> Output mapping (this instruction is to be removed)
          auto next_inst_result_0 = next_inst.result().front();
          auto next_inst_result_1 = next_inst.result().back();

          auto new_inst_result_0 = new_inst.result().front();
          auto new_inst_result_1 = new_inst.result().back();

          next_inst_result_0.replaceAllUsesWith(new_inst_result_0);
          next_inst_result_1.replaceAllUsesWith(new_inst_result_1);
        } else {
          // Zero rotation: just map from input -> output
          auto next_inst_result_0 = next_inst.result().front();
          auto next_inst_result_1 = next_inst.result().back();
          next_inst_result_0.replaceAllUsesWith(op.getOperand(0));
          next_inst_result_1.replaceAllUsesWith(op.getOperand(1));
        }
      } else {
        // Need to create an AddFOp
        auto add_op = rewriter.create<mlir::AddFOp>(op.getLoc(), first_angle,
                                                    second_angle);
        assert(add_op.result().getType().isa<mlir::FloatType>());

        auto new_inst = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
            op.getLoc(), llvm::makeArrayRef(ret_types), result_inst_name,
            llvm::makeArrayRef({op.getOperand(0), op.getOperand(1)}),
            llvm::makeArrayRef({add_op.result()}));
        // Input -> Output mapping (this instruction is to be removed)
        auto next_inst_result_0 = next_inst.result().front();
        auto next_inst_result_1 = next_inst.result().back();

        auto new_inst_result_0 = new_inst.result().front();
        auto new_inst_result_1 = new_inst.result().back();

        next_inst_result_0.replaceAllUsesWith(new_inst_result_0);
        next_inst_result_1.replaceAllUsesWith(new_inst_result_1);
      }

      // Cache instructions for delete.
      deadOps.emplace_back(op);
      deadOps.emplace_back(next_inst);
    }
  });

  for (auto &op : deadOps) {
    op->dropAllUses();
    op.erase();
  }
}
} // namespace qcor
