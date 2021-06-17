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
    for (;;) {
      // Break inside:
      if (current_op.qubits().size() > 1) {
        break;
      }
      auto return_value = *current_op.result().begin();
      if (return_value.hasOneUse()) {
        if (current_op.getNumOperands() > 1) {
          // Parameterized gate:
          std::vector<double> current_op_params;
          for (size_t i = 1; i < current_op.getNumOperands(); ++i) {
            auto param = current_op.getOperand(i);
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
                break;
              }
            } else {
              std::cout << "Get non-constant param. Stop.\n";
              break;
            }
          }
          op_params.emplace_back(current_op_params);
        } else {
          op_params.emplace_back(std::vector<double>{});
        }
        ops_list.emplace_back(current_op);
        // get that one user
        auto user = *return_value.user_begin();
        if (auto next_inst =
                dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(user)) {
          current_op = next_inst;
        } else {
          break;
        }
      } else {
        break;
      }
    }

    std::cout << "Find sequence of: " << ops_list.size() << "\n";
    constexpr int MIN_SEQ_LENGTH = 2;
    if (ops_list.size() > MIN_SEQ_LENGTH) {
      // Should try to optimize:
      // TODO:
    }

    return failure();
  }
};
} // namespace qcor
