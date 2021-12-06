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
#include "ConditionalOpLowering.hpp"
#include <iostream>

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

namespace qcor {
LogicalResult ConditionalOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  auto location = parentModule->getLoc();
  auto context = parentModule->getContext();
  auto if_op = cast<mlir::quantum::ConditionalOp>(op);
  auto callable_type =
      LLVM::LLVMPointerType::get(get_quantum_type("Callable", context));
  auto result_type =
      LLVM::LLVMPointerType::get(get_quantum_type("Result", context));
  FlatSymbolRefAttr qir_symbol_ref;
  if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_apply_if_else_op)) {
    qir_symbol_ref = SymbolRefAttr::get(qir_apply_if_else_op, context);
  } else {
    // Signature:
    // void __quantum__qis__applyifelseintrinsic__body(Result *r,
    //                                             Callable *clb_on_zero,
    //                                             Callable *clb_on_one);
    auto apply_ifelse_ftype = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(context),
        llvm::ArrayRef<Type>{result_type, callable_type, callable_type}, false);

    // Insert the function declaration
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(parentModule.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(parentModule->getLoc(),
                                      qir_apply_if_else_op, apply_ifelse_ftype);
    qir_symbol_ref = mlir::SymbolRefAttr::get(qir_apply_if_else_op, context);
  }

  // We don't support else yet, hence always null
  mlir::Value callable_nullPtr =
      rewriter.create<LLVM::NullOp>(location, callable_type);
  rewriter.create<mlir::CallOp>(
      location, qir_symbol_ref, LLVM::LLVMVoidType::get(context),
      ArrayRef<Value>(
          {if_op.result_bit(), callable_nullPtr, if_op.then_callable()}));
  rewriter.eraseOp(op);

  return success();
}
} // namespace qcor