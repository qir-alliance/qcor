#pragma once

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "quantum_to_llvm.hpp"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

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
                  ConversionPatternRewriter &rewriter) const override {
    // Local Declarations, get location, parentModule
    // and the context
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();

    // First step is to get a reference to the Symbol Reference for the
    // qalloc QIR runtime function, this will only declare it once and reuse
    // each time it is seen
    FlatSymbolRefAttr symbol_ref;
    if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(
            qir_qubit_array_deallocate)) {
      symbol_ref = SymbolRefAttr::get(qir_qubit_array_deallocate, context);
    } else {
      // prototype is (Array*) -> void
      auto void_type = LLVM::LLVMVoidType::get(context);
      auto array_qbit_type =
          LLVM::LLVMPointerType::get(get_quantum_type("Array", context));
      auto dealloc_ftype =
          LLVM::LLVMFunctionType::get(void_type, array_qbit_type, false);

      // Insert the function declaration
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(
          parentModule->getLoc(), qir_qubit_array_deallocate, dealloc_ftype);
      symbol_ref =
          mlir::SymbolRefAttr::get(qir_qubit_array_deallocate, context);
    }

    // Get as a QallocOp, get its allocatino size and qreg variable name
    auto deallocOp = cast<mlir::quantum::DeallocOp>(op);
    auto qubits_value = deallocOp.qubits();
    auto qreg_name_attr = qubits_value.getDefiningOp()->getAttr("name");
    auto name = qreg_name_attr.cast<::mlir::StringAttr>().getValue();
    auto qubits = variables[name.str()];

    // create a CallOp for the new quantum runtime de-allocation
    // function.
    rewriter.create<mlir::CallOp>(loc, symbol_ref,
                                  LLVM::LLVMVoidType::get(context),
                                  ArrayRef<Value>({qubits}));

    // Remove the old QuantumDialect QallocOp
    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace qcor