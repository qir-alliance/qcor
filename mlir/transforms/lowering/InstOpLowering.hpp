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
// The goal of InstOpLowering is to convert all QuantumDialect
// InstOp (quantum.inst) to the corresponding __quantum__qis__INST(int64*, ...)
// call
class InstOpLowering : public ConversionPattern {
protected:
  // Symbol table, local seen variables
  std::map<std::string, mlir::Value> &variables;

  // Mapping of Vector::ExtractElementOp Operation pointers to the
  // corresponding qreg variable name they represent
  std::map<mlir::Operation *, std::string> &qubit_extract_map;

  std::vector<std::string> &module_function_names;
  mutable std::map<std::string, std::string> inst_map{{"cx", "cnot"},
                                                      {"measure", "mz"}};

public:
  // The Constructor, store the variables and qubit extract op map
  explicit InstOpLowering(MLIRContext *context,
                          std::map<std::string, mlir::Value> &vars,
                          std::map<mlir::Operation *, std::string> &qem,
                          std::vector<std::string> &f_names)
      : ConversionPattern(mlir::quantum::InstOp::getOperationName(), 1,
                          context),
        variables(vars), qubit_extract_map(qem),
        module_function_names(f_names) {}

  // Match and replace all InstOps
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

// Lower Result type casting:
// In QCOR QIR runtime, Result is just a bool (i1)
// hence, just need to do a type cast and load.
class ResultCastOpLowering : public ConversionPattern {
protected:
public:
  explicit ResultCastOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::ResultCastOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

class IntegerCastOpLowering : public ConversionPattern {
protected:
public:
  explicit IntegerCastOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::IntegerCastOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace qcor