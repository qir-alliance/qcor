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
// The goal of QubitArrayAllocOpLowering is to lower all occurrences of the
// MLIR QuantumDialect createQubitArray to the MSFT QIR
// __quantum__rt__array_create_1d() quantum runtime function for Qubit*
// (create a generic array holding references to Qubit for aliasing purposes)
// as an LLVM MLIR Function and CallOp.
class QubitArrayAllocOpLowering : public ConversionPattern {
protected:
  // Constant string for runtime function name
  inline static const std::string qir_qubit_array_allocate =
      "__quantum__rt__array_create_1d";
  // Rudimentary symbol table, seen variables
  std::map<std::string, mlir::Value> &variables;
  /// Lower to:
  /// %Array* @__quantum__rt__array_create_1d(i32 %elementSizeInBytes, i64%
  /// nQubits) where elementSizeInBytes = 8 (pointer size).
public:
  // Constructor, store seen variables
  explicit QubitArrayAllocOpLowering(MLIRContext *context,
                                     std::map<std::string, mlir::Value> &vars)
      : ConversionPattern(mlir::quantum::QaliasArrayAllocOp::getOperationName(),
                          1, context),
        variables(vars) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace qcor