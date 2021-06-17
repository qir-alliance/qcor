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
          std::cout << "Combine " << next_inst.name().str() << " and "
                    << inst_name.str() << "\n";
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
          auto add_op = rewriter.create<mlir::AddFOp>(op.getLoc(), first_angle,
                                                      second_angle);
          assert(add_op.result().getType().isa<mlir::FloatType>());
          // Create a new instruction:
          // Return type: qubit
          std::vector<mlir::Type> ret_types{op.getOperand(0).getType()};
          const std::string result_inst_name = (inst_name.str()[0] == 'r')
                                                   ? inst_name.str()
                                                   : next_inst.name().str();
          assert(result_inst_name[0] == 'r');
          auto new_inst = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
              op.getLoc(), llvm::makeArrayRef(ret_types), result_inst_name,
              llvm::makeArrayRef(op.getOperand(0)),
              llvm::makeArrayRef({add_op.result()}));

          std::cout << "AFTER MERGE ANGLE:\n";
          auto parentModule = op->getParentOfType<mlir::ModuleOp>();
          parentModule->dump();

          // Input -> Output mapping (this instruction is to be removed)
          (*next_inst.result_begin())
              .replaceAllUsesWith(*new_inst.result_begin());
          // Erase both original instructions:
          rewriter.eraseOp(op);
          rewriter.eraseOp(next_inst);
          std::cout << "AFTER REMOVE:\n";
          parentModule->dump();

          // Done
          return success();
        }
      }
    }
    return failure();
  }
};
} // namespace qcor
