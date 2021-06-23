#include "SingleQubitGateMergingPass.hpp"
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
#include "utils/gate_matrix.hpp"
#include <iostream>

namespace qcor {
void SingleQubitGateMergingPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}

void SingleQubitGateMergingPass::runOnOperation() {
  // Walk the operations within the function.
  std::vector<mlir::quantum::ValueSemanticsInstOp> deadOps;
  getOperation().walk([&](mlir::quantum::ValueSemanticsInstOp op) {
    if (std::find(deadOps.begin(), deadOps.end(), op) != deadOps.end()) {
      // Skip this op since it was merged (forward search)
      return;
    }
    mlir::OpBuilder rewriter(op);

    // List of ops:
    std::vector<mlir::quantum::ValueSemanticsInstOp> ops_list;
    std::vector<std::vector<double>> op_params;
    auto current_op = op;

    // Helper to retrieve VSOp constant params:
    const auto retrieveConstantGateParams =
        [](mlir::quantum::ValueSemanticsInstOp &vs_op) -> std::vector<double> {
      if (vs_op.getNumOperands() > 1) {
        // Parameterized gate:
        std::vector<double> current_op_params;
        for (size_t i = 1; i < vs_op.getNumOperands(); ++i) {
          auto param = vs_op.getOperand(i);
          assert(param.getType().isa<mlir::FloatType>());
          auto def_op = param.getDefiningOp();
          if (def_op) {
            if (auto const_def_op =
                    dyn_cast_or_null<mlir::ConstantFloatOp>(def_op)) {
              llvm::APFloat param_const_val = const_def_op.getValue();
              const double param_val = param_const_val.convertToDouble();
              current_op_params.emplace_back(param_val);
              // std::cout << "Get constant param: " << param_val << "\n";
            } else {
              // std::cout << "Get non-constant param. Stop.\n";
              return {};
            }
          } else {
            // std::cout << "Cannot locate the defining op. Stop.\n";
            return {};
          }
        }
        return current_op_params;
      } else {
        return {};
      }
    };

    for (;;) {
      // Break inside:
      if (current_op.qubits().size() > 1) {
        break;
      }
      auto return_value = *current_op.result().begin();
      if (return_value.hasOneUse()) {
        const auto const_op_params = retrieveConstantGateParams(current_op);
        // Params are not constant:
        if (const_op_params.size() < current_op.getNumOperands() - 1) {
          break;
        }

        ops_list.emplace_back(current_op);
        op_params.emplace_back(const_op_params);
        // get that one user
        auto user = *return_value.user_begin();
        if (auto next_inst =
                dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(user)) {
          current_op = next_inst;
        } else {
          break;
        }
      } else {
        if (current_op.qubits().size() == 1) {
          const auto const_op_params = retrieveConstantGateParams(current_op);
          // Params are not constant:
          if (const_op_params.size() < current_op.getNumOperands() - 1) {
            break;
          }

          ops_list.emplace_back(current_op);
          op_params.emplace_back(const_op_params);
        }
        break;
      }
    }

    assert(ops_list.size() == op_params.size());
    constexpr int MIN_SEQ_LENGTH = 2;
    if (ops_list.size() > MIN_SEQ_LENGTH) {
      // Should try to optimize:
      std::vector<qcor::utils::qop_t> ops;
      for (size_t i = 0; i < ops_list.size(); ++i) {
        ops.emplace_back(
            std::make_pair(ops_list[i].name().str(), op_params[i]));
      }
      const auto simplified_seq = qcor::utils::decompose_gate_sequence(ops);

      if (simplified_seq.size() < ops_list.size()) {
        // std::cout << "Find simpler gate sequence\n";
        rewriter.setInsertionPointAfter(ops_list.back());
        std::vector<mlir::quantum::ValueSemanticsInstOp> new_ops;
        for (const auto &[pauli_inst, theta] : simplified_seq) {
          mlir::Value theta_val = rewriter.create<mlir::ConstantOp>(
              op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), theta));
          std::vector<mlir::Type> ret_types{op.getOperand(0).getType()};
          auto new_inst = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
              op.getLoc(), llvm::makeArrayRef(ret_types), pauli_inst,
              llvm::makeArrayRef(new_ops.empty()
                                     ? op.getOperand(0)
                                     : *(new_ops.back().result_begin())),
              llvm::makeArrayRef({theta_val}));
          new_ops.emplace_back(new_inst);
        }

        // Input -> Output mapping (this instruction is to be removed)
        auto last_inst_orig = ops_list.back();
        auto last_inst_new = new_ops.back();
        (*last_inst_orig.result_begin())
            .replaceAllUsesWith(*last_inst_new.result_begin());

        // Erase original instructions:
        for (auto &op_to_delete : ops_list) {
          deadOps.emplace_back(op_to_delete);
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