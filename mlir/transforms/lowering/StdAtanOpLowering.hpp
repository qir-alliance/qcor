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
class StdAtanOpLowering : public ConversionPattern {
private:
  static FlatSymbolRefAttr getOrInsertAtanFunction(PatternRewriter &rewriter,
                                                   ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("atan"))
      return mlir::SymbolRefAttr::get("atan", context);

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto ret_type = rewriter.getF64Type();
    auto arg_type = rewriter.getF64Type();
    auto llvmFnType = LLVM::LLVMFunctionType::get(ret_type, arg_type, false);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "atan", llvmFnType);
    return mlir::SymbolRefAttr::get("atan", context);
  }

public:
  // Constructor, store seen variables
  explicit StdAtanOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::AtanOp::getOperationName(), 1, context) {}

  // Match any Operation that is the QallocOp
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Local Declarations, get location, parentModule
    // and the context
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto atan = cast<mlir::AtanOp>(op);

    auto atanRef = getOrInsertAtanFunction(rewriter, parentModule);

    auto call = rewriter.create<mlir::LLVM::CallOp>(loc, rewriter.getF64Type(),
                                                    atanRef, atan.operand());

    rewriter.replaceOp(op, call.getResult(0));

    return success();
  }
};
} // namespace qcor