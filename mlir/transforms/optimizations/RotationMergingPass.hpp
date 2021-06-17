#include "Quantum/QuantumOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace qcor {

class RotationMergingPattern
    : public mlir::OpRewritePattern<mlir::quantum::ValueSemanticsInstOp> {
protected:
  // Use a simple list of pairs to handle bi-directional check.
  // This list is quite short so performance shouldn't be an issue!
  const std::vector<std::pair<std::string, std::string>> search_gates{
      {"rx", "rx"}, {"ry", "ry"}, {"rz", "rz"},
      {"x", "rx"},  {"y", "ry"},  {"z", "rz"}};
  bool should_combine(const std::string &name1,
                      const std::string &name2) const {
    return std::find_if(
               search_gates.begin(), search_gates.end(),
               [&](const std::pair<std::string, std::string> &gate_pair) {
                 return (gate_pair.first == name1 &&
                         gate_pair.second == name2) ||
                        (gate_pair.first == name2 && gate_pair.second == name1);
               }) != search_gates.end();
  }

public:
  RotationMergingPattern(mlir::MLIRContext *context)
      : OpRewritePattern<mlir::quantum::ValueSemanticsInstOp>(context,
                                                              /*benefit=*/10) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::quantum::ValueSemanticsInstOp op,
                  mlir::PatternRewriter &rewriter) const override {
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
          std::cout << "Combine " << next_inst.name().str() << " and " << inst_name.str() << "\n";
          // Determine which of the two is the rotation gate:
          // we can combine rx with x (which is rx(pi)) as well
          if (inst_name.str()[0] == 'r') {
            // Merge to the first instruction
            // Replace all uses of second with its input.
            auto first_angle = op.getOperand(1);
            assert(first_angle.getType().isa<mlir::FloatType>());
            if (next_inst.name().str()[0] == 'r') {
              // Merge two rotations
              // TODO: this is currently not working...
              auto second_angle = next_inst.getOperand(1);
              assert(second_angle.getType().isa<mlir::FloatType>());
              auto add_op = rewriter.create<mlir::AddFOp>(
                  next_inst.getLoc(), *(first_angle.getDefiningOp()->result_begin()),
                  *(second_angle.getDefiningOp()->result_begin()));
              // next_inst.getOperand(1).replaceAllUsesWith(add_op.result());
              next_inst.setOperand(1, add_op.result());

              std::cout << "AFTER MERGE ANGLE:\n";
              auto parentModule = op->getParentOfType<mlir::ModuleOp>();
              parentModule->dump();

              // Input -> Output mapping (this instruction is to be removed)
              // Note: we keep the second rotation in the pair
              (*op.result_begin()).replaceAllUsesWith(op.getOperand(0));
              rewriter.eraseOp(op);

              std::cout << "AFTER REMOVE:\n";
              parentModule->dump();
            } else {
              // Merge theta with PI:
              mlir::Value pi_val = rewriter.create<mlir::ConstantOp>(
                  op.getLoc(),
                  mlir::FloatAttr::get(rewriter.getF64Type(), M_PI));
              // std::cout << "Define op:\n";
              // op.getOperand(1).getDefiningOp()->dump();
              auto add_op = rewriter.create<mlir::AddFOp>(
                  op.getLoc(), pi_val,
                  *(op.getOperand(1).getDefiningOp()->result_begin()));
              assert(add_op.result().getType().isa<mlir::FloatType>());
              // op.getOperand(1).replaceAllUsesWith(add_op.result());
              op.setOperand(1, add_op.result());
              std::cout << "AFTER MERGE ANGLE:\n";
              auto parentModule = op->getParentOfType<mlir::ModuleOp>();
              parentModule->dump();

              // Input -> Output mapping (this instruction is to be removed)
              (*next_inst.result_begin())
                  .replaceAllUsesWith(next_inst.getOperand(0));
              rewriter.eraseOp(next_inst);

              std::cout << "AFTER REMOVE:\n";
              parentModule->dump();
            }

            return success();
          }
          else {
            // TODO:
            // Merge first (x,y,z) to second rotation
            // Pass input of op => next op
            (*op.result_begin()).replaceAllUsesWith(op.getOperand(0));
            rewriter.eraseOp(op);
          }
          
          return success();
        }
      }
    }

    return failure();
  }
};
} // namespace qcor
