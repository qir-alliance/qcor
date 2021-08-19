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
      tuple_struct_type_list.push_back(
          LLVM::LLVMPointerType::get(get_quantum_type("Array", context)));
    } else if (result.getType().isa<mlir::OpaqueType>() &&
               result.getType().cast<mlir::OpaqueType>().getTypeData() ==
                   "Qubit") {
      tuple_struct_type_list.push_back(
          LLVM::LLVMPointerType::get(get_quantum_type("Qubit", context)));
    } else if (result.getType().isa<mlir::OpaqueType>() &&
               result.getType().cast<mlir::OpaqueType>().getTypeData() ==
                   "Tuple") {
      tuple_struct_type_list.push_back(
          LLVM::LLVMPointerType::get(get_quantum_type("Tuple", context)));
    } else if (result.getType().isa<mlir::FloatType>()) {
      tuple_struct_type_list.push_back(mlir::FloatType::getF64(context));
    } else if (result.getType().isa<mlir::IntegerType>()) {
      tuple_struct_type_list.push_back(mlir::IntegerType::get(context, 64));
    } else {
      std::cout << "WE DON'T SUPPORT TUPLE UNPACK FOR THE TYPE\n";
      exit(0);
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

  const std::string kernel_name = create_callable_op.functors().str();
  const std::string BODY_WRAPPER_NAME = kernel_name + "__body__wrapper";
  const std::string ADJOINT_WRAPPER_NAME = kernel_name + "__adj__wrapper";
  const std::string CTRL_WRAPPER_NAME = kernel_name + "__ctl__wrapper";
  const std::string CTRL_ADJOINT_WRAPPER_NAME =
      kernel_name + "__ctladj__wrapper";

  const std::vector<mlir::Value> functor_ptr_values{
      // Base
      rewriter.create<LLVM::AddressOfOp>(
          location, LLVM::LLVMPointerType::get(callable_entry_ftype),
          SymbolRefAttr::get(BODY_WRAPPER_NAME.c_str(), context)),
      // Adjoint
      rewriter.create<LLVM::AddressOfOp>(
          location, LLVM::LLVMPointerType::get(callable_entry_ftype),
          SymbolRefAttr::get(ADJOINT_WRAPPER_NAME.c_str(), context)),
      // Controlled
      rewriter.create<LLVM::AddressOfOp>(
          location, LLVM::LLVMPointerType::get(callable_entry_ftype),
          SymbolRefAttr::get(CTRL_WRAPPER_NAME.c_str(), context)),
      // Controlled Adjoint
      rewriter.create<LLVM::AddressOfOp>(
          location, LLVM::LLVMPointerType::get(callable_entry_ftype),
          SymbolRefAttr::get(CTRL_ADJOINT_WRAPPER_NAME.c_str(), context)),
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
        llvm::ArrayRef<Type>{
            LLVM::LLVMPointerType::get(callable_entry_fn_array_type),
            LLVM::LLVMPointerType::get(callback_fn_array_type), tuple_type},
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
  mlir::Value capture_tuple_ptr = [&]() {
    if (create_callable_op.captures().empty()) {
      auto op = rewriter.create<LLVM::NullOp>(location, tuple_type);
      return op.res();
    } else {
      mlir::SmallVector<mlir::Type> tuple_struct_type_list;
      size_t tuple_size_in_bytes = 0;
      for (const auto &captured_var : create_callable_op.captures()) {
        if (captured_var.getType().isa<mlir::OpaqueType>() &&
            captured_var.getType().cast<mlir::OpaqueType>().getTypeData() ==
                "Array") {
          tuple_struct_type_list.push_back(
              LLVM::LLVMPointerType::get(get_quantum_type("Array", context)));
          tuple_size_in_bytes += sizeof(void *);
        } else if (captured_var.getType().isa<mlir::OpaqueType>() &&
                   captured_var.getType()
                           .cast<mlir::OpaqueType>()
                           .getTypeData() == "Qubit") {
          tuple_struct_type_list.push_back(
              LLVM::LLVMPointerType::get(get_quantum_type("Qubit", context)));
          tuple_size_in_bytes += sizeof(void *);
        } else if (captured_var.getType().isa<mlir::OpaqueType>() &&
                   captured_var.getType()
                           .cast<mlir::OpaqueType>()
                           .getTypeData() == "Tuple") {
          tuple_struct_type_list.push_back(
              LLVM::LLVMPointerType::get(get_quantum_type("Tuple", context)));
          tuple_size_in_bytes += sizeof(void *);
        } else if (captured_var.getType().isa<mlir::FloatType>()) {
          tuple_struct_type_list.push_back(mlir::FloatType::getF64(context));
          tuple_size_in_bytes += sizeof(double);
        } else if (captured_var.getType().isa<mlir::IntegerType>()) {
          tuple_struct_type_list.push_back(mlir::IntegerType::get(context, 64));
          tuple_size_in_bytes += sizeof(int64_t);
        } else {
          std::cout << "WE DON'T SUPPORT TUPLE PACK FOR THE TYPE\n";
          exit(0);
        }
      }
      mlir::Value tuple_size_value = rewriter.create<LLVM::ConstantOp>(
          location, mlir::IntegerType::get(rewriter.getContext(), 64),
          rewriter.getIntegerAttr(
              mlir::IntegerType::get(rewriter.getContext(), 64),
              tuple_size_in_bytes));

      // Tuple create signature: TuplePtr __quantum__rt__tuple_create(int64_t
      // size)
      FlatSymbolRefAttr tuple_create_symbol_ref = [&]() {
        const std::string qir_tuple_create_fn_name =
            "__quantum__rt__tuple_create";
        if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(
                qir_tuple_create_fn_name)) {
          return SymbolRefAttr::get(qir_tuple_create_fn_name, context);
        } else {
          auto ftype = LLVM::LLVMFunctionType::get(
              tuple_type,
              llvm::ArrayRef<Type>{mlir::IntegerType::get(context, 64)}, false);

          // Insert the function declaration
          PatternRewriter::InsertionGuard insertGuard(rewriter);
          rewriter.setInsertionPointToStart(parentModule.getBody());
          rewriter.create<LLVM::LLVMFuncOp>(parentModule->getLoc(),
                                            qir_tuple_create_fn_name, ftype);
          return mlir::SymbolRefAttr::get(qir_tuple_create_fn_name, context);
        }
      }();
      auto createTupleCallOp = rewriter.create<mlir::CallOp>(
          location, tuple_create_symbol_ref, tuple_type,
          ArrayRef<Value>({tuple_size_value}));
      mlir::Value tuplePtr = createTupleCallOp.getResult(0);

      // Store to tuple:
      auto tuple_struct_type =
          LLVM::LLVMStructType::getLiteral(context, tuple_struct_type_list);
      auto structPtr =
          rewriter
              .create<LLVM::BitcastOp>(
                  location, LLVM::LLVMPointerType::get(tuple_struct_type),
                  tuplePtr)
              .res();

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
        auto store_op = rewriter.create<LLVM::StoreOp>(
            location, create_callable_op.captures()[idx], field_ptr);
      }

      return tuplePtr;
    }
  }();
  auto createCallableCallOp = rewriter.create<mlir::CallOp>(
      location, qir_symbol_ref, callable_return_type,
      ArrayRef<Value>(
          {callable_entry_fn_array, callbacks_nullPtr, capture_tuple_ptr}));
  rewriter.replaceOp(op, createCallableCallOp.getResult(0));
  return success();
}
} // namespace qcor