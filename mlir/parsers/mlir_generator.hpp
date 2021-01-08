#pragma once
#include <string>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

namespace qcor {
class QuantumMLIRGenerator {
 protected:
  mlir::MLIRContext& context;
  mlir::ModuleOp m_module;
  mlir::OpBuilder builder;
  mlir::Block* main_entry_block;

 public:
  QuantumMLIRGenerator(mlir::MLIRContext& ctx) : context(ctx), builder(&ctx) {}

  // This method can be implemented by subclasses to
  // introduce any initialization steps required for constructing
  // mlir using the quantum dialect. This may also be used for
  // introducing any initialization operations before
  // generation of the rest of the mlir code.
  virtual void initialize_mlirgen() = 0;

  // This method can be implemented by subclasses to map a
  // quantum code in a subclass-specific source language to
  // the internal generated MLIR ModuleOp instance
  virtual void mlirgen(const std::string& src) = 0;

  // Return the generated ModuleOp
  mlir::OwningModuleRef get_module() {
    return mlir::OwningModuleRef(mlir::OwningOpRef<mlir::ModuleOp>(m_module));
  }

  // Finalize method, override to provide any end operations
  // to the module (like a return_op).
  virtual void finalize_mlirgen() = 0;
};
}  // namespace qcor