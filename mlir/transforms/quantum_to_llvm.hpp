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

#include "Quantum/QuantumOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace qcor {

struct QuantumToLLVMLoweringPass
    : public PassWrapper<QuantumToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override;
  void runOnOperation() final;

 private:
  bool q_optimizations = false;
  std::vector<std::string>& function_names;

 public:
  QuantumToLLVMLoweringPass(std::vector<std::string>& f_names)
      : function_names(f_names) {}
  QuantumToLLVMLoweringPass(bool q_opts, std::vector<std::string>& f_names)
      : q_optimizations(q_opts), function_names(f_names) {}
};

// Helper func.
mlir::Type get_quantum_type(std::string type, mlir::MLIRContext* context);

// std::unique_ptr<mlir::Pass> createQuantumOptPass();
}  // namespace qcor