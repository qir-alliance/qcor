#include "CreateStringLiteralOpLowering.hpp"
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
/// Return a value representing an access into a global string with the given
/// name, creating the string if necessary.
Value CreateStringLiteralOpLowering::getOrCreateGlobalString(Location loc,
                                                             OpBuilder &builder,
                                                             StringRef name,
                                                             StringRef value,
                                                             ModuleOp module) {
  // Create the global at the entry of the module.
  LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMArrayType::get(
        IntegerType::get(builder.getContext(), 8), value.size());
    global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                            LLVM::Linkage::Internal, name,
                                            builder.getStringAttr(value));
  }

  // Get the pointer to the first character in the global string.
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value cst0 = builder.create<LLVM::ConstantOp>(
      loc, IntegerType::get(builder.getContext(), 64),
      builder.getIntegerAttr(builder.getIndexType(), 0));
  return builder.create<LLVM::GEPOp>(
      loc,
      LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
      globalPtr, ArrayRef<Value>({cst0, cst0}));
}

// Match any Operation that is the QallocOp
LogicalResult CreateStringLiteralOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // Local Declarations, get location, parentModule
  // and the context
  auto loc = op->getLoc();
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  auto slOp = cast<mlir::quantum::CreateStringLiteralOp>(op);
  auto slOpText = slOp.text();
  auto slVarName = slOp.varname();

  Value new_global_str = getOrCreateGlobalString(
      loc, rewriter, slVarName,
      StringRef(slOpText.str().c_str(), slOpText.str().length() + 1),
      parentModule);

  // The string literal var must be **overriden** by closest scope.
  // This will prevent dangling references b/w different scopes leading to
  // dominance checking failed.
  // Notes: the above getOrCreateGlobalString will just get a *reference*
  // to the globally-allocated string.
  // i.e., each scope must use its own reference (potentially to the same
  // string). Otherwise, we'll have dominance check failure.
  variables.insert_or_assign(slVarName.str(), new_global_str);

  rewriter.eraseOp(op);

  return success();
}
} // namespace qcor