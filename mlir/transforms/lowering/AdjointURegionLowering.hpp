#pragma once
#include "quantum_to_llvm.hpp"

namespace qcor {
class StartAdjointURegionOpLowering : public ConversionPattern {
public:
  // CTor: store seen variables
  explicit StartAdjointURegionOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::StartAdjointURegion::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
class EndAdjointURegionOpLowering : public ConversionPattern {
public:
  // CTor: store seen variables
  explicit EndAdjointURegionOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::EndAdjointURegion::getOperationName(), 1,
                          context){}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace qcor