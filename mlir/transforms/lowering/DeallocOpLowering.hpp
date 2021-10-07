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
// declare void @__quantum__rt__qubit_release_array(%Array*)
class DeallocOpLowering : public ConversionPattern {
protected:
  // Constant string for runtime function name
  inline static const std::string qir_qubit_array_deallocate =
      "__quantum__rt__qubit_release_array";
  // Rudimentary symbol table, seen variables
  std::map<std::string, mlir::Value> &variables;

  // %Array* @__quantum__rt__qubit_allocate_array(i64 %nQubits)
public:
  // Constructor, store seen variables
  explicit DeallocOpLowering(MLIRContext *context,
                             std::map<std::string, mlir::Value> &vars)
      : ConversionPattern(mlir::quantum::DeallocOp::getOperationName(), 1,
                          context),
        variables(vars) {}

  // Match any Operation that is the QallocOp
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace qcor