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
      location, IntegerType::get(rewriter.getContext(), 64),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
  for (size_t idx = 0; idx < tuple_struct_type_list.size(); ++idx) {
    mlir::Value idx_cst = rewriter.create<LLVM::ConstantOp>(
        location, IntegerType::get(rewriter.getContext(), 64),
        rewriter.getIntegerAttr(rewriter.getIndexType(), idx));
    auto field_ptr =
        rewriter
            .create<LLVM::GEPOp>(
                location,
                LLVM::LLVMPointerType::get(tuple_struct_type_list[idx]),
                structPtr, mlir::ArrayRef<mlir::Value>({zero_cst, idx_cst}))
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
  return success();
}
} // namespace qcor