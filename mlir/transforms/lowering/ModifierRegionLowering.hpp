#pragma once
#include "quantum_to_llvm.hpp"

namespace qcor {
class PowURegionOpLowering : public ConversionPattern {
public:
  explicit PowURegionOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::PowURegion::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

class CtrlURegionOpLowering : public ConversionPattern {
public:
  explicit CtrlURegionOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::CtrlURegion::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

class AdjURegionOpLowering : public ConversionPattern {
public:
  explicit AdjURegionOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::AdjURegion::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

class EndModifierRegionOpLowering : public ConversionPattern {
public:
  explicit EndModifierRegionOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::ModifierEndOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ModifierRegionRewritePass
    : public PassWrapper<ModifierRegionRewritePass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() final;
  ModifierRegionRewritePass() {}
};
} // namespace qcor