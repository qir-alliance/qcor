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
#include "QubitArrayAllocOpLowering.hpp"
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
LogicalResult QubitArrayAllocOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // Local Declarations, get location, parentModule
  // and the context
  auto loc = op->getLoc();
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  auto context = parentModule->getContext();

  // First step is to get a reference to the Symbol Reference for the
  // __quantum__rt__array_create_1d QIR runtime function,
  // this will only declare it once and reuse each time it is seen
  FlatSymbolRefAttr symbol_ref;
  if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_qubit_array_allocate)) {
    symbol_ref = SymbolRefAttr::get(qir_qubit_array_allocate, context);
  } else {
    // prototype is (elementSize: int32, arraySize : int64) -> Array* :
    // qubit_array_ptr
    auto qubit_type = IntegerType::get(context, 64);
    auto element_size_type = IntegerType::get(context, 32);
    auto array_qbit_type =
        LLVM::LLVMPointerType::get(get_quantum_type("Array", context));
    auto array_alloc_ftype = LLVM::LLVMFunctionType::get(
        array_qbit_type, llvm::ArrayRef<Type>{element_size_type, qubit_type},
        false);

    // Insert the function declaration
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(parentModule.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(
        parentModule->getLoc(), qir_qubit_array_allocate, array_alloc_ftype);
    symbol_ref = mlir::SymbolRefAttr::get(qir_qubit_array_allocate, context);
  }

  // Get as a QaliasArrayAllocOp, get its allocation size and qreg variable
  // name
  auto qallocOp = cast<mlir::quantum::QaliasArrayAllocOp>(op);
  auto size = qallocOp.size();
  auto qreg_name = qallocOp.name().str();

  Value create_size_int = rewriter.create<LLVM::ConstantOp>(
      loc, IntegerType::get(rewriter.getContext(), 64),
      rewriter.getIntegerAttr(rewriter.getI64Type(), size));

  Value element_size_int = rewriter.create<LLVM::ConstantOp>(
      loc, IntegerType::get(rewriter.getContext(), 32),
      rewriter.getIntegerAttr(
          rewriter.getI64Type(),
          /* element size = pointer size */ sizeof(void *)));

  auto array_qbit_type =
      LLVM::LLVMPointerType::get(get_quantum_type("Array", context));
  auto qalloc_qir_call = rewriter.create<mlir::CallOp>(
      loc, symbol_ref, array_qbit_type,
      ArrayRef<Value>({element_size_int, create_size_int}));

  // Get the returned qubit array pointer Value
  auto qbit_array = qalloc_qir_call.getResult(0);

  // Remove the old QuantumDialect QallocOp
  rewriter.replaceOp(op, qbit_array);
  // Save the qubit array variable to the symbol table
  variables.insert({qreg_name, qbit_array});
  // std::cout << "Array 1D alloc:\n";
  // parentModule.dump();
  return success();
}
} // namespace qcor