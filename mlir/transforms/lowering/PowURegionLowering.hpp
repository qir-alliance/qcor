#pragma once
#include "quantum_to_llvm.hpp"

namespace qcor {
class StartPowURegionOpLowering : public ConversionPattern {
public:
  // CTor: store seen variables
  explicit StartPowURegionOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::StartPowURegion::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
class EndPowURegionOpLowering : public ConversionPattern {
public:
  // CTor: store seen variables
  explicit EndPowURegionOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::EndPowURegion::getOperationName(), 1,
                          context){}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace qcor