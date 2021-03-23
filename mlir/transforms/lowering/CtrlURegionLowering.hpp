#pragma once
#include "quantum_to_llvm.hpp"

namespace qcor {
class StartCtrlURegionOpLowering : public ConversionPattern {
 public:
  // CTor: store seen variables
  explicit StartCtrlURegionOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::StartCtrlURegion::getOperationName(),
                          1, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};
class EndCtrlURegionOpLowering : public ConversionPattern {
 public:
  // CTor: store seen variables
  explicit EndCtrlURegionOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::EndCtrlURegion::getOperationName(), 1,
                          context) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};
}  // namespace qcor