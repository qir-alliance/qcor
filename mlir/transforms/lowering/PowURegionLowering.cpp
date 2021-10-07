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
#include "PowURegionLowering.hpp"

#include <iostream>

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
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

namespace qcor {
LogicalResult StartPowURegionOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // Local Declarations
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  auto context = parentModule->getContext();
  auto location = parentModule->getLoc();

  FlatSymbolRefAttr qir_get_fn_ptr = [&]() {
    static const std::string qir_start_func =
        "__quantum__rt__start_pow_u_region";
    if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_start_func)) {
      return SymbolRefAttr::get(qir_start_func, context);
    } else {
      
      // prototype should be (int64_t) -> void :

      auto void_type = LLVM::LLVMVoidType::get(context);

      auto func_type =
          LLVM::LLVMFunctionType::get(void_type, llvm::ArrayRef<Type>{}, false);

      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(location, qir_start_func, func_type);

      return mlir::SymbolRefAttr::get(qir_start_func, context);
    }
  }();

  rewriter.create<mlir::CallOp>(location, qir_get_fn_ptr,
                                LLVM::LLVMVoidType::get(context),
                                llvm::makeArrayRef(std::vector<mlir::Value>{}));

  rewriter.eraseOp(op);

  return success();
}

LogicalResult EndPowURegionOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // Local Declarations
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  auto context = parentModule->getContext();
  auto location = parentModule->getLoc();

  FlatSymbolRefAttr qir_get_fn_ptr = [&]() {
    static const std::string qir_end_func = "__quantum__rt__end_pow_u_region";
    if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_end_func)) {
      return SymbolRefAttr::get(qir_end_func, context);
    } else {
      // prototype should be (int64) -> void :

      auto void_type = LLVM::LLVMVoidType::get(context);

      auto func_type = LLVM::LLVMFunctionType::get(
          void_type, llvm::ArrayRef<Type>{rewriter.getI64Type()}, false);

      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(location, qir_end_func, func_type);

      return mlir::SymbolRefAttr::get(qir_end_func, context);
    }
  }();

  // Get the power
  // auto endpow = cast<mlir::quantum::EndPowURegion>(op);
  // auto pow = endpow.pow();

  // Value create_pow_int = rewriter.create<LLVM::ConstantOp>(
  //     location, IntegerType::get(rewriter.getContext(), 64),
  //     rewriter.getIntegerAttr(rewriter.getI64Type(), pow));

  rewriter.create<mlir::CallOp>(
      location, qir_get_fn_ptr, LLVM::LLVMVoidType::get(context),
      operands);

  rewriter.eraseOp(op);

  return success();
}
}  // namespace qcor