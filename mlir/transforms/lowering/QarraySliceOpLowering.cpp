#include "QarraySliceOpLowering.hpp"
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
LogicalResult QarraySliceOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op->getLoc();
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  auto context = parentModule->getContext();
  // %Range = type { i64, i64, i64 }
  auto range_type = LLVM::LLVMStructType::getIdentified(context, "Range");
  range_type.setBody(llvm::ArrayRef<Type>{IntegerType::get(context, 64),
                                          IntegerType::get(context, 64),
                                          IntegerType::get(context, 64)},
                     false);
  auto array_qbit_type =
      LLVM::LLVMPointerType::get(get_quantum_type("Array", context));
  FlatSymbolRefAttr symbol_ref;
  if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_qubit_array_slice)) {
    symbol_ref = SymbolRefAttr::get(qir_qubit_array_slice, context);
  } else {
    // prototype is (%Array*, i32, %Range) -> %Array*
    auto qslice_ftype = LLVM::LLVMFunctionType::get(
        array_qbit_type,
        llvm::ArrayRef<Type>{array_qbit_type, IntegerType::get(context, 32),
                             range_type},
        false);

    // Insert the function declaration
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(parentModule.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(parentModule->getLoc(),
                                      qir_qubit_array_slice, qslice_ftype);
    symbol_ref = mlir::SymbolRefAttr::get(qir_qubit_array_slice, context);
  }

  // Get as a QarraySliceOp
  auto qArraySliceOp = cast<mlir::quantum::ArraySliceOp>(op);
  auto input_array = qArraySliceOp.qreg();
  auto slice_range = qArraySliceOp.slice_range();
  assert(slice_range.size() == 3);
  // Create a Range object:
  auto rangeObj = [&]() {
    auto desc = rewriter.create<LLVM::UndefOp>(loc, range_type);
    auto insertStart = rewriter.create<LLVM::InsertValueOp>(
        loc, desc, slice_range[0], rewriter.getI64ArrayAttr(0));
    auto insertStep = rewriter.create<LLVM::InsertValueOp>(
        loc, insertStart, slice_range[1], rewriter.getI64ArrayAttr(1));
    auto insertEnd = rewriter.create<LLVM::InsertValueOp>(
        loc, insertStep, slice_range[2], rewriter.getI64ArrayAttr(2));

    return insertEnd.res();
  }();

  // Retrieve the input array
  auto qreg_name_attr = input_array.getDefiningOp()->getAttr("name");
  auto name = qreg_name_attr.cast<::mlir::StringAttr>().getValue();
  auto array_var = variables[name.str()];
  // create a CallOp for the new quantum runtime allocation
  // (%Array*, i32, %Range) -> %Array*
  Value dim_id_int = rewriter.create<LLVM::ConstantOp>(
      loc, IntegerType::get(rewriter.getContext(), 32),
      rewriter.getIntegerAttr(rewriter.getI64Type(),
                              /* dim = 0, 1d */ 0));

  // Make the call
  auto slice_array_call = rewriter.create<mlir::CallOp>(
      loc, symbol_ref, array_qbit_type,
      ArrayRef<Value>({array_var, dim_id_int, rangeObj}));

  variables.insert({name.str(), slice_array_call.getResult(0)});

  // Remove the old QuantumDialect QarraySliceOp
  rewriter.replaceOp(op, slice_array_call.getResult(0));

  return success();
}
} // namespace qcor