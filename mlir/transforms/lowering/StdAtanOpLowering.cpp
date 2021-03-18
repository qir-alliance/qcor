#include "StdAtanOpLowering.hpp"
#include <iostream>

namespace qcor {
FlatSymbolRefAttr
StdAtanOpLowering::getOrInsertAtanFunction(PatternRewriter &rewriter,
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

// Match any Operation that is the QallocOp
LogicalResult
StdAtanOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                   ConversionPatternRewriter &rewriter) const {
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
} // namespace qcor