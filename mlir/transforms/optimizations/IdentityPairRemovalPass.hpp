#include "Quantum/QuantumOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace qcor {

class SingleQubitIdentityPairRemovalPattern
    : public mlir::OpRewritePattern<mlir::quantum::ValueSemanticsInstOp> {
 protected:
  const std::map<std::string, std::string> search_gates{
      {"x", "x"}, {"y", "y"}, {"z", "z"}, {"h", "h"},
      {"t", "tdg"}, {"tdg", "t"}, {"s", "sdg"}, {"sdg", "s"}};
  bool should_remove(std::string name1, std::string name2) const {
    if (search_gates.count(name1)) {
      return search_gates.at(name1) == name2;
    }
    return false;
  }

 public:
  SingleQubitIdentityPairRemovalPattern(mlir::MLIRContext* context)
      : OpRewritePattern<mlir::quantum::ValueSemanticsInstOp>(context,
                                                              /*benefit=*/10) {}
  mlir::LogicalResult matchAndRewrite(
      mlir::quantum::ValueSemanticsInstOp op,
      mlir::PatternRewriter& rewriter) const override {

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

          rewriter.eraseOp(op);
          rewriter.eraseOp(next_inst);

          return success();
        }
      }
    }

    return failure();
  }
};

class CNOTIdentityPairRemovalPattern
    : public mlir::OpRewritePattern<mlir::quantum::ValueSemanticsInstOp> {
 protected:
  const std::map<std::string, std::string> search_gates{
      {"x", "x"}, {"y", "y"}, {"z", "z"}, {"h", "h"},
      {"t", "tdg"}, {"tdg", "t"}, {"s", "sdg"}, {"sdg", "s"}};
  bool should_remove(std::string name1, std::string name2) const {
    if (search_gates.count(name1)) {
      return search_gates.at(name1) == name2;
    }
    return false;
  }

 public:
  CNOTIdentityPairRemovalPattern(mlir::MLIRContext* context)
      : OpRewritePattern<mlir::quantum::ValueSemanticsInstOp>(context,
                                                              /*benefit=*/10) {}
  mlir::LogicalResult matchAndRewrite(
      mlir::quantum::ValueSemanticsInstOp op,
      mlir::PatternRewriter& rewriter) const override {

    auto inst_name = op.name();
    if (inst_name != "cnot" && inst_name != "cx") {
      return failure();
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
      auto next_inst = dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(src_user);
      auto tmp_tgt = dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(tgt_user);
      if (!next_inst || !tmp_tgt) {
        // not inst ops
        return failure();
      }

      // We want the case where src_user and tgt_user are the same
      if (next_inst.getOperation() != tmp_tgt.getOperation()) {
        return failure();
      }

      // Need src_return_val to map to next_inst operand 0, 
      // and tgt_return_val to map to next_inst operand 1. 
      // if not drop out
      if (next_inst.getOperand(0) != src_return_val && next_inst.getOperand(1) != tgt_return_val) {
        return failure();
      }  
      
      // They are the same operation, a cnot
      // so we have cnot src, tgt | cnot src, tgt
      auto next_inst_result_0 = next_inst.result().front();
      auto next_inst_result_1 = next_inst.result().back();
      next_inst_result_0.replaceAllUsesWith(op.getOperand(0));
      next_inst_result_1.replaceAllUsesWith(op.getOperand(1));

      // Remove the identity pair
      rewriter.eraseOp(op);
      rewriter.eraseOp(src_user);

      return success();

    }
    
    return failure();
  }
};

}  // namespace qcor
