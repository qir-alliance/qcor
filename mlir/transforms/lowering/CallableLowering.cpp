#include "CallableLowering.hpp"
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
LogicalResult TupleUnpackOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  assert(operands.size() == 1);
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  auto context = parentModule->getContext();
  // Cast the tuple to a struct type:
  auto tuple_unpack_op = cast<mlir::quantum::TupleUnpackOp>(op);
  mlir::SmallVector<mlir::Type> tuple_struct_type_list;
  for (const auto &result : tuple_unpack_op.result()) {
    if (result.getType().isa<mlir::OpaqueType>() &&
        result.getType().cast<mlir::OpaqueType>().getTypeData() == "Array") {
      auto array_type =
          LLVM::LLVMPointerType::get(get_quantum_type("Array", context));
      tuple_struct_type_list.push_back(array_type);
    } else if (result.getType().isa<mlir::FloatType>()) {
      auto float_type = mlir::FloatType::getF64(context);
      tuple_struct_type_list.push_back(float_type);
    }
  }

  auto unpacked_struct_type =
      LLVM::LLVMStructType::getLiteral(context, tuple_struct_type_list);
  auto location = parentModule->getLoc();
  auto structPtr =
      rewriter
          .create<LLVM::BitcastOp>(
              location, LLVM::LLVMPointerType::get(unpacked_struct_type),
              operands[0])
          .res();

  mlir::SmallVector<mlir::Value> unpacked_vals;
  mlir::Value zero_cst = rewriter.create<LLVM::ConstantOp>(
      location, IntegerType::get(rewriter.getContext(), 32),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
  for (size_t idx = 0; idx < tuple_struct_type_list.size(); ++idx) {
    mlir::Value idx_cst = rewriter.create<LLVM::ConstantOp>(
        location, IntegerType::get(rewriter.getContext(), 32),
        rewriter.getIntegerAttr(rewriter.getIndexType(), idx));
    auto field_ptr =
        rewriter
            .create<LLVM::GEPOp>(
                location,
                LLVM::LLVMPointerType::get(tuple_struct_type_list[idx]),
                structPtr, ArrayRef<Value>({zero_cst, idx_cst}))
            .res();
    auto load_op = rewriter.create<LLVM::LoadOp>(
        location, tuple_struct_type_list[idx], field_ptr);
    unpacked_vals.push_back(load_op.res());
  }

  rewriter.replaceOp(op, unpacked_vals);
  // std::cout << "After:\n";
  // parentModule.dump();
  return success();
}

LogicalResult CreateCallableOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  auto location = parentModule->getLoc();
  auto context = parentModule->getContext();
  auto create_callable_op = cast<mlir::quantum::CreateCallableOp>(op);  
  // Signature: void (%Tuple*, %Tuple*, %Tuple*)
  auto tuple_type =
      LLVM::LLVMPointerType::get(get_quantum_type("Tuple", context));
  // typedef void (*CallableEntryType)(TuplePtr, TuplePtr, TuplePtr);
  // typedef void (*CaptureCallbackType)(TuplePtr, int32_t);
  auto callable_entry_ftype = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(context),
      llvm::ArrayRef<Type>{tuple_type, tuple_type, tuple_type}, false);
  auto capture_callback_ftype = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(context),
      llvm::ArrayRef<Type>{tuple_type,
                           IntegerType::get(rewriter.getContext(), 32)},
      false);
  FlatSymbolRefAttr symbol_ref =
      SymbolRefAttr::get(create_callable_op.functors(), context);
  auto callable_entry_fn_array_type = LLVM::LLVMArrayType::get(
      LLVM::LLVMPointerType::get(callable_entry_ftype), 4);
  auto callback_fn_array_type = LLVM::LLVMArrayType::get(
      LLVM::LLVMPointerType::get(capture_callback_ftype), 2);
  const std::string functor_array_name =
      create_callable_op.functors().str() + "__Qops";
  mlir::Value value_1_const = rewriter.create<LLVM::ConstantOp>(
      location, IntegerType::get(rewriter.getContext(), 64),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
  mlir::Value callable_entry_fn_array = rewriter.create<LLVM::AllocaOp>(
      location, LLVM::LLVMPointerType::get(callable_entry_fn_array_type),
      value_1_const,
      /*alignment=*/0);

  const std::vector<mlir::Value> functor_ptr_values{
      // Base
      rewriter.create<LLVM::AddressOfOp>(
          location, LLVM::LLVMPointerType::get(callable_entry_ftype),
          symbol_ref),
      // Adjoint
      rewriter.create<LLVM::NullOp>(
          location, LLVM::LLVMPointerType::get(callable_entry_ftype)),
      // Controlled
      rewriter.create<LLVM::NullOp>(
          location, LLVM::LLVMPointerType::get(callable_entry_ftype)),
      // Controlled Adjoint
      rewriter.create<LLVM::NullOp>(
          location, LLVM::LLVMPointerType::get(callable_entry_ftype)),
  };

  mlir::Value zero_index = rewriter.create<LLVM::ConstantOp>(
      location, IntegerType::get(rewriter.getContext(), 64),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
  for (size_t func_idx = 0; func_idx < functor_ptr_values.size(); ++func_idx) {
    mlir::Value func_index_val = rewriter.create<LLVM::ConstantOp>(
        location, IntegerType::get(rewriter.getContext(), 64),
        rewriter.getIntegerAttr(rewriter.getIndexType(), func_idx));
    mlir::Value func_in_array_ptr = rewriter.create<LLVM::GEPOp>(
        location,
        LLVM::LLVMPointerType::get(
            LLVM::LLVMPointerType::get(callable_entry_ftype)),
        callable_entry_fn_array, ArrayRef<Value>({zero_index, func_index_val}));
    rewriter.create<LLVM::StoreOp>(location, functor_ptr_values[func_idx],
                                   func_in_array_ptr);
  }

  auto callable_return_type =
      LLVM::LLVMPointerType::get(get_quantum_type("Callable", context));
  FlatSymbolRefAttr qir_symbol_ref;
  if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_create_callable)) {
    qir_symbol_ref = SymbolRefAttr::get(qir_create_callable, context);
  } else {
    // Callable *
    // __quantum__rt__callable_create(Callable::CallableEntryType *ft,
    //                           Callable::CaptureCallbackType *callbacks,
    //                           TuplePtr capture)
    auto create_callable_ftype = LLVM::LLVMFunctionType::get(
        callable_return_type,
        llvm::ArrayRef<Type>{LLVM::LLVMPointerType::get(callable_entry_fn_array_type),
                             LLVM::LLVMPointerType::get(callback_fn_array_type),
                             tuple_type},
        false);

    // Insert the function declaration
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(parentModule.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(
        parentModule->getLoc(), qir_create_callable, create_callable_ftype);
    qir_symbol_ref = mlir::SymbolRefAttr::get(qir_create_callable, context);
  }
  
  // Callbacks and captured tuple ==> null
  mlir::Value callbacks_nullPtr = rewriter.create<LLVM::NullOp>(
      location, LLVM::LLVMPointerType::get(callback_fn_array_type));
  mlir::Value tuple_nullPtr =
      rewriter.create<LLVM::NullOp>(location, tuple_type);
  auto createCallableCallOp = rewriter.create<mlir::CallOp>(
      location, qir_symbol_ref, callable_return_type,
      ArrayRef<Value>(
          {callable_entry_fn_array, callbacks_nullPtr, tuple_nullPtr}));
  rewriter.replaceOp(op, createCallableCallOp.getResult(0));
  return success();
}
} // namespace qcor