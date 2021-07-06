#pragma once
#include "quantum_to_llvm.hpp"

namespace qcor {
class TupleUnpackOpLowering : public ConversionPattern {
protected:
public:
  explicit TupleUnpackOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::TupleUnpackOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

class CreateCallableOpLowering : public ConversionPattern {
protected:
public:
  explicit CreateCallableOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::CreateCallableOp::getOperationName(),
                          1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace qcor