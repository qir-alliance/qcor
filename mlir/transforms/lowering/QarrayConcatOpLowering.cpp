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
#include "QarrayConcatOpLowering.hpp"
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
LogicalResult QarrayConcatOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op->getLoc();
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  auto context = parentModule->getContext();
  auto array_qbit_type =
      LLVM::LLVMPointerType::get(get_quantum_type("Array", context));
  FlatSymbolRefAttr symbol_ref;
  if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_qubit_array_concat)) {
    symbol_ref = SymbolRefAttr::get(qir_qubit_array_concat, context);
  } else {
    // prototype is (%Array*, %Array*) -> %Array*
    auto qconcat_ftype = LLVM::LLVMFunctionType::get(
        array_qbit_type, llvm::ArrayRef<Type>{array_qbit_type, array_qbit_type},
        false);

    // Insert the function declaration
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(parentModule.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(parentModule->getLoc(),
                                      qir_qubit_array_concat, qconcat_ftype);
    symbol_ref = mlir::SymbolRefAttr::get(qir_qubit_array_concat, context);
  }

  // Make the call
  auto array_concat_call =
      rewriter.create<mlir::CallOp>(loc, symbol_ref, array_qbit_type, operands);
  // Remove the old QuantumDialect QarrayConcatOp
  rewriter.replaceOp(op, array_concat_call.getResult(0));

  return success();
}
} // namespace qcor