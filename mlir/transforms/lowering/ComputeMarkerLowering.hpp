#pragma once
#include "quantum_to_llvm.hpp"

namespace qcor {
class ComputeMarkerOpLowering : public ConversionPattern {
public:
  // CTor: store seen variables
  explicit ComputeMarkerOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::ComputeMarkerOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
class ComputeUnMarkerOpLowering : public ConversionPattern {
public:
  // CTor: store seen variables
  explicit ComputeUnMarkerOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::ComputeUnMarkerOp::getOperationName(), 1,
                          context){}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace qcor