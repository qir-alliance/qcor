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
  std::cout << "Before:\n";
  parentModule.dump();
  // Cast the tuple to a struct type:
  auto tuple_unpack_op = cast<mlir::quantum::TupleUnpackOp>(op);
  std::vector<Type> unpacked_type_list;
  std::vector<Type> tuple_struct_type_list;
  for (const auto &result : tuple_unpack_op.result()) {
    if (result.getType().isa<mlir::OpaqueType>() &&
        result.getType().cast<mlir::OpaqueType>().getTypeData() == "Array") {
      auto array_type =
          LLVM::LLVMPointerType::get(get_quantum_type("Array", context));
      unpacked_type_list.emplace_back(LLVM::LLVMPointerType::get(array_type));
      tuple_struct_type_list.emplace_back(array_type);
    } else if (result.getType().isa<mlir::FloatType>()) {
      auto float_type = mlir::FloatType::getF64(context);
      unpacked_type_list.emplace_back(LLVM::LLVMPointerType::get(float_type));
      tuple_struct_type_list.emplace_back(float_type);
    } 
  }

  auto unpacked_type = LLVM::LLVMStructType::getLiteral(
      context, llvm::ArrayRef<Type>(tuple_struct_type_list));
  auto location = parentModule->getLoc();
  auto bitcast = rewriter.create<LLVM::BitcastOp>(
      location, LLVM::LLVMPointerType::get(unpacked_type), operands[0]);

  std::vector<mlir::Value> unpacked_vals;
  for (size_t idx = 0; idx < unpacked_type_list.size(); ++idx) {
    mlir::Value idx_cst = rewriter.create<LLVM::ConstantOp>(
        location, IntegerType::get(rewriter.getContext(), 64),
        rewriter.getIntegerAttr(rewriter.getIndexType(), idx));
    auto getelementptr = rewriter.create<LLVM::GEPOp>(
        location, unpacked_type_list[idx], bitcast,
        idx_cst);
    auto load_op = rewriter.create<LLVM::LoadOp>(
        location, tuple_struct_type_list[idx], getelementptr.res());
    unpacked_vals.emplace_back(load_op.res());
  }

  for (size_t idx = 0; idx < unpacked_vals.size(); ++idx) {
    mlir::Value unpack_result = *(std::next(tuple_unpack_op.result_begin(), idx));
    unpack_result.replaceAllUsesWith(unpacked_vals[idx]);
  }
  rewriter.eraseOp(op);
  std::cout << "After:\n";
  parentModule.dump();
  return success();
}

LogicalResult CreateCallableOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  return success();
}
} // namespace qcor