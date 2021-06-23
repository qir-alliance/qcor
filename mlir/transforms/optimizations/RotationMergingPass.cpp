#include "RotationMergingPass.hpp"
#include "Quantum/QuantumOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <iostream>

namespace qcor {
bool RotationMergingPass::should_combine(const std::string &name1,
                                         const std::string &name2) const {
  return std::find_if(
             search_gates.begin(), search_gates.end(),
             [&](const std::pair<std::string, std::string> &gate_pair) {
               return (gate_pair.first == name1 && gate_pair.second == name2) ||
                      (gate_pair.first == name2 && gate_pair.second == name1);
             }) != search_gates.end();
}

void RotationMergingPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}

void RotationMergingPass::runOnOperation() {
  // Walk the operations within the function.
  std::vector<mlir::quantum::ValueSemanticsInstOp> deadOps;
  getOperation().walk([&](mlir::quantum::ValueSemanticsInstOp op) {
    if (std::find(deadOps.begin(), deadOps.end(), op) != deadOps.end()) {
      // Skip this op since it was merged (forward search)
      return;
    }
    mlir::OpBuilder rewriter(op);
    auto inst_name = op.name();
    auto return_value = *op.result().begin();
    if (return_value.hasOneUse()) {
      // get that one user
      auto user = *return_value.user_begin();
      // cast to a inst op
      if (auto next_inst =
              dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(user)) {
        // check that we can merge these two gates
        if (should_combine(next_inst.name().str(), inst_name.str())) {
          // std::cout << "Combine " << next_inst.name().str() << " and "
          //           << inst_name.str() << "\n";
          // Determine which of the two is the rotation gate:
          // we can combine rx with x (which is rx(pi)) as well
          mlir::Value pi_val = rewriter.create<mlir::ConstantOp>(
              op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), M_PI));

          // Angle = theta (if is a rotation gate); or PI (Pauli gate)
          mlir::Value first_angle =
              (inst_name.str()[0] == 'r') ? op.getOperand(1) : pi_val;
          mlir::Value second_angle = (next_inst.name().str()[0] == 'r')
                                         ? next_inst.getOperand(1)
                                         : pi_val;
          // Must set insertion point so that the Add op
          // is placed **after** the second instruction,
          // e.g. to be after the second theta angle definition.
          rewriter.setInsertionPointAfter(next_inst);
          // Try retrieve the angle if possible (constant).
          // So that we compute the total angle here as well.
          // This will make sure that all compile-time constants are propagated
          // b/w MLIR passes.
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

          const auto first_angle_const = tryGetConstAngle(first_angle);
          const auto second_angle_const = tryGetConstAngle(first_angle);

          // Create a new instruction:
          // Return type: qubit
          std::vector<mlir::Type> ret_types{op.getOperand(0).getType()};
          const std::string result_inst_name = (inst_name.str()[0] == 'r')
                                                   ? inst_name.str()
                                                   : next_inst.name().str();
          assert(result_inst_name[0] == 'r');
          if (first_angle_const.has_value() && second_angle_const.has_value()) {
            // both angles are constant: pre-compute the total angle:
            const double totalAngle =
                first_angle_const.value() + second_angle_const.value();
            mlir::Value totalAngleVal = rewriter.create<mlir::ConstantOp>(
                op.getLoc(),
                mlir::FloatAttr::get(rewriter.getF64Type(), totalAngle));
            auto new_inst =
                rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                    op.getLoc(), llvm::makeArrayRef(ret_types),
                    result_inst_name, llvm::makeArrayRef(op.getOperand(0)),
                    llvm::makeArrayRef({totalAngleVal}));
            // Input -> Output mapping (this instruction is to be removed)
            (*next_inst.result_begin())
                .replaceAllUsesWith(*new_inst.result_begin());
          } else {
            // Need to create an AddFOp
            auto add_op = rewriter.create<mlir::AddFOp>(
                op.getLoc(), first_angle, second_angle);
            assert(add_op.result().getType().isa<mlir::FloatType>());

            auto new_inst =
                rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                    op.getLoc(), llvm::makeArrayRef(ret_types),
                    result_inst_name, llvm::makeArrayRef(op.getOperand(0)),
                    llvm::makeArrayRef({add_op.result()}));
            // Input -> Output mapping (this instruction is to be removed)
            (*next_inst.result_begin())
                .replaceAllUsesWith(*new_inst.result_begin());
          }

          // Cache instructions for delete.
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
} // namespace qcor
