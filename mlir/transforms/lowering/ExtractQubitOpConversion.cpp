#include "ExtractQubitOpConversion.hpp"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
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
LogicalResult ExtractQubitOpConversion::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // Local Declarations
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  auto context = parentModule->getContext();
  auto location = parentModule->getLoc();

  // First goal, get symbol for
  // %0 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %q, i64 0)
  // %1 = bitcast i8* %0 to %Qubit**
  // %.qb = load %Qubit*, %Qubit** %1
  FlatSymbolRefAttr symbol_ref;
  if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_get_qubit_from_array)) {
    symbol_ref = SymbolRefAttr::get(qir_get_qubit_from_array, context);
  } else {
    // prototype should be (int64* : qreg, int64 : element) -> int64* : qubit
    auto qubit_array_type =
        LLVM::LLVMPointerType::get(get_quantum_type("Array", context));
    auto qubit_index_type = IntegerType::get(context, 64);
    // ret is i8*
    auto qbit_element_ptr_type =
        LLVM::LLVMPointerType::get(IntegerType::get(context, 8));

    auto get_ptr_qbit_ftype = LLVM::LLVMFunctionType::get(
        qbit_element_ptr_type,
        llvm::ArrayRef<Type>{qubit_array_type, qubit_index_type}, false);

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(parentModule.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(location, qir_get_qubit_from_array,
                                      get_ptr_qbit_ftype);

    symbol_ref = mlir::SymbolRefAttr::get(qir_get_qubit_from_array, context);
  }

  // Create the CallOp for the get element ptr 1d function
  auto array_qbit_type =
      LLVM::LLVMPointerType::get(IntegerType::get(context, 8));

  auto get_qbit_qir_call = rewriter.create<mlir::CallOp>(
      location, symbol_ref, array_qbit_type, operands);

  auto bitcast = rewriter.create<LLVM::BitcastOp>(
      location,
      LLVM::LLVMPointerType::get(
          LLVM::LLVMPointerType::get(get_quantum_type("Qubit", context))),
      get_qbit_qir_call.getResult(0));
  auto real_casted_qubit = rewriter.create<LLVM::LoadOp>(
      location, LLVM::LLVMPointerType::get(get_quantum_type("Qubit", context)),
      bitcast.res());

  rewriter.replaceOp(op, real_casted_qubit.res());

  return success();
}
} // namespace qcor