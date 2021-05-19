#include "Quantum/QuantumOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace qcor {

class RemoveUnusedExtractQubitCalls
    : public mlir::OpRewritePattern<mlir::quantum::ExtractQubitOp> {
 public:
  RemoveUnusedExtractQubitCalls(mlir::MLIRContext* context)
      : OpRewritePattern<mlir::quantum::ExtractQubitOp>(context,
                                                        /*benefit=*/10) {}
  mlir::LogicalResult matchAndRewrite(
      mlir::quantum::ExtractQubitOp op,
      mlir::PatternRewriter& rewriter) const override {
    // Qubits returned from q.extract have exactly 
    // one user, the next ValueSemanticsInstOp. If 
    // it does not have exactly one, then it has 0, so 
    // lets remove it as the value is not even used.
    auto qubit = op.qbit();
    if (!qubit.hasOneUse()) {
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

class RemoveUnusedQallocCalls
    : public mlir::OpRewritePattern<mlir::quantum::QallocOp> {
 public:
  RemoveUnusedQallocCalls(mlir::MLIRContext* context)
      : OpRewritePattern<mlir::quantum::QallocOp>(context,
                                                  /*benefit=*/10) {}
  mlir::LogicalResult matchAndRewrite(
      mlir::quantum::QallocOp op,
      mlir::PatternRewriter& rewriter) const override {

    // QallocOp returns a qubit array, if it 
    // only has one user, then that user must be 
    // the dealloc op, so we can remove both of them
    auto qubits = op.qubits();
        auto next_op = (*qubits.user_begin());
    if (dyn_cast_or_null<mlir::quantum::DeallocOp>(next_op) &&
            qubits.hasOneUse()) {
      rewriter.eraseOp(op);
      rewriter.eraseOp(next_op);
      return success();
    }

    return failure();
  }
};

}  // namespace qcor
