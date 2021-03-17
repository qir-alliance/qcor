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
// The goal of QallocOpLowering is to lower all occurrences of the
// MLIR QuantumDialect QallocOp to the MSFT QIR
// __quantum__rt__qubit_allocate_array() quantum runtime function as an LLVM
// MLIR Function and CallOp.
class QallocOpLowering : public ConversionPattern {
protected:
  // Constant string for runtime function name
  inline static const std::string qir_qubit_array_allocate =
      "__quantum__rt__qubit_allocate_array";
  // Rudimentary symbol table, seen variables
  std::map<std::string, mlir::Value> &variables;

  // %Array* @__quantum__rt__qubit_allocate_array(i64 %nQubits)
public:
  // Constructor, store seen variables
  explicit QallocOpLowering(MLIRContext *context,
                            std::map<std::string, mlir::Value> &vars)
      : ConversionPattern(mlir::quantum::QallocOp::getOperationName(), 1,
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
    if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_qubit_array_allocate)) {
      symbol_ref = SymbolRefAttr::get(qir_qubit_array_allocate, context);
    } else {
      // prototype is (size : int64) -> Array* : qubit_array_ptr
      auto qubit_type = IntegerType::get(context, 64);
      auto array_qbit_type =
          LLVM::LLVMPointerType::get(get_quantum_type("Array", context));
      auto qalloc_ftype =
          LLVM::LLVMFunctionType::get(array_qbit_type, qubit_type, false);

      // Insert the function declaration
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(parentModule->getLoc(),
                                        qir_qubit_array_allocate, qalloc_ftype);
      symbol_ref = mlir::SymbolRefAttr::get(qir_qubit_array_allocate, context);
    }

    // Get as a QallocOp, get its allocatino size and qreg variable name
    auto qallocOp = cast<mlir::quantum::QallocOp>(op);
    auto size = qallocOp.size();
    auto qreg_name = qallocOp.name().str();

    // create a CallOp for the new quantum runtime allocation
    // function.
    // size_value = constantop (size)
    // qubit_array_ptr = callop ( size_value )
    Value create_size_int = rewriter.create<LLVM::ConstantOp>(
        loc, IntegerType::get(rewriter.getContext(), 64),
        rewriter.getIntegerAttr(rewriter.getI64Type(), size));
    auto array_qbit_type =
        LLVM::LLVMPointerType::get(get_quantum_type("Array", context));
    auto qalloc_qir_call = rewriter.create<mlir::CallOp>(
        loc, symbol_ref, array_qbit_type, ArrayRef<Value>({create_size_int}));

    // Get the returned qubit array pointer Value
    auto qbit_array = qalloc_qir_call.getResult(0);

    // Remove the old QuantumDialect QallocOp
    rewriter.replaceOp(op, qbit_array);
    rewriter.eraseOp(op);
    // Save the qubit array variable to the symbol table
    variables.insert({qreg_name, qbit_array});

    return success();
  }
};
} // namespace qcor