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
#include <string>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include <map>

namespace qcor {
class QuantumMLIRGenerator {
 protected:
  mlir::MLIRContext& context;
  mlir::ModuleOp m_module;
  mlir::OpBuilder builder;
  mlir::Block* main_entry_block;
  std::vector<std::string> function_names;

 public:
  QuantumMLIRGenerator(mlir::MLIRContext& ctx) : context(ctx), builder(&ctx) {}
  QuantumMLIRGenerator(mlir::OpBuilder b, mlir::MLIRContext& ctx)
      : context(ctx), builder(b) {}

  // This method can be implemented by subclasses to
  // introduce any initialization steps required for constructing
  // mlir using the quantum dialect. This may also be used for
  // introducing any initialization operations before
  // generation of the rest of the mlir code.
  virtual void initialize_mlirgen(bool add_entry_point = true,
                                  const std::string file_name = "", std::map<std::string, std::string> extra_quantum_args = {}) = 0;

  // This method can be implemented by subclasses to map a
  // quantum code in a subclass-specific source language to
  // the internal generated MLIR ModuleOp instance
  virtual void mlirgen(const std::string& src) = 0;

  // Return the generated ModuleOp
  mlir::OwningModuleRef get_module() {
    return mlir::OwningModuleRef(mlir::OwningOpRef<mlir::ModuleOp>(m_module));
  }
  
  std::vector<std::string> seen_function_names() { return function_names; }

  // Finalize method, override to provide any end operations
  // to the module (like a return_op).
  virtual void finalize_mlirgen() = 0;
};
}  // namespace qcor