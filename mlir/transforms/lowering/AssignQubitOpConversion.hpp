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
class AssignQubitOpConversion : public ConversionPattern {
protected:
  std::map<std::string, mlir::Value> &variables;

public:
  // CTor: store seen variables
  explicit AssignQubitOpConversion(MLIRContext *context,
                                   std::map<std::string, mlir::Value> &vars)
      : ConversionPattern(mlir::quantum::AssignQubitOp::getOperationName(), 1,
                          context),
        variables(vars) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace qcor