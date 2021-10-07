/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
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