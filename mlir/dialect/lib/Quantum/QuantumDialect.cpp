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
#include "Quantum/QuantumDialect.h"
#include "Quantum/QuantumOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::quantum;
namespace {
/// Inliner interface
/// This class defines the interface for handling inlining with Quantum
/// operations.
// We simplify inherit from the base interface class and override
/// the necessary methods.
struct QuantumInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// This hook checks to see if the given callable operation is legal to inline
  /// into the given call.
  /// Operations in Quantum dialect are always legal to inline.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// This hook checks to see if the given operation is legal to inline into the
  /// given region.
  /// Only inline VSOp for now:
  // FIXME: there is a weird error when qalloc is inlined at MLIR level
  // hence, just allow VSOp to be inlined for the timebeing.
  // i.e. all quantum subroutines that only contain VSOp's can be inlined.
  bool isLegalToInline(Operation *op, Region *region, bool,
                       BlockAndValueMapping &) const final {
    if (dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(op) ||
        dyn_cast_or_null<mlir::quantum::ExtractQubitOp>(op)) {
      return true;
    }

    return false;
  }
};
} // namespace
//===----------------------------------------------------------------------===//
// Quantum dialect.
//===----------------------------------------------------------------------===//

void QuantumDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Quantum/QuantumOps.cpp.inc"
      >();
  addInterfaces<QuantumInlinerInterface>();
}

// static void print(mlir::OpAsmPrinter &printer, mlir::quantum::InstOp op) {
//   printer << "q." << op.name() << "(" << *(op.qubits().begin());
//   for (auto i = 1; i < op.qubits().size(); i++) {
//     printer << ", " << op.qubits()[i];
//   }
//   printer << ")";

// }

void PowURegion::build(OpBuilder &builder, OperationState &result, Value pow, ValueRange qubits) {
  assert(pow.getType().isIntOrIndex());
  result.addOperands(pow);
  result.addOperands(qubits);
  std::vector<Type> resultTypes;
  for (const auto &qubitOperand : qubits) {
    resultTypes.emplace_back(qubitOperand.getType());
  }
  result.addTypes(resultTypes);
  OpBuilder::InsertionGuard guard(builder);
  Region *body = result.addRegion();
  builder.createBlock(body);
}

void AdjURegion::build(OpBuilder &builder, OperationState &result, ValueRange qubits) {
  result.addOperands(qubits);
  std::vector<Type> resultTypes;
  for (const auto &qubitOperand : qubits) {
    resultTypes.emplace_back(qubitOperand.getType());
  }
  result.addTypes(resultTypes);
  OpBuilder::InsertionGuard guard(builder);
  Region *body = result.addRegion();
  builder.createBlock(body);
}

void CtrlURegion::build(OpBuilder &builder, OperationState &result, Value ctrl_bit, ValueRange qubits) {
  assert(ctrl_bit.getType().isa<mlir::OpaqueType>());
  result.addOperands(ctrl_bit);
  result.addOperands(qubits);
  std::vector<Type> resultTypes;
  resultTypes.emplace_back(ctrl_bit.getType());
  for (const auto &qubitOperand : qubits) {
    resultTypes.emplace_back(qubitOperand.getType());
  }
  result.addTypes(resultTypes);
  OpBuilder::InsertionGuard guard(builder);
  Region *body = result.addRegion();
  builder.createBlock(body);
}
