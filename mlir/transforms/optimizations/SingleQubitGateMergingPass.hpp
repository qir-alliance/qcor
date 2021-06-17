#include "Quantum/QuantumOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace qcor {
// Try to merge a sequence of single-qubit ops:
// This should be placed after simple pattern-matching methods
// (no need to compute the matrix)
class SingleQubitGateMergingPattern
    : public mlir::OpRewritePattern<mlir::quantum::ValueSemanticsInstOp> {
public:
  SingleQubitGateMergingPattern(mlir::MLIRContext *context)
      : OpRewritePattern<mlir::quantum::ValueSemanticsInstOp>(context,
                                                              /*benefit=*/10) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::quantum::ValueSemanticsInstOp op,
                  mlir::PatternRewriter &rewriter) const override {
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
              std::cout << "Get constant param: " << param_val << "\n";
            } else {
              std::cout << "Get non-constant param. Stop.\n";
              return {};
            }
          } else {
            std::cout << "Cannot locate the defining op. Stop.\n";
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

    std::cout << "Find sequence of: " << ops_list.size() << "\n";
    for (auto &op : ops_list) {
      std::cout << op.name().str() << " ";
    }
    std::cout << "\n";
    assert(ops_list.size() == op_params.size());
    constexpr int MIN_SEQ_LENGTH = 2;
    if (ops_list.size() > MIN_SEQ_LENGTH) {
      // Should try to optimize:
      // TODO:
    }

    return failure();
  }
};
} // namespace qcor
