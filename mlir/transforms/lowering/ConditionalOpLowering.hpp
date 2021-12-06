/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#pragma once
#include "quantum_to_llvm.hpp"
namespace qcor {
class ConditionalOpLowering : public ConversionPattern {
protected:
public:
  inline static const std::string qir_apply_if_else_op =
      "__quantum__qis__applyifelseintrinsic__body";
  explicit ConditionalOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::ConditionalOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace qcor