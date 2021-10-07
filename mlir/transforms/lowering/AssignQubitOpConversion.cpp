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
#include "AssignQubitOpConversion.hpp"
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
LogicalResult AssignQubitOpConversion::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // Local Declarations
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  auto context = parentModule->getContext();
  auto location = parentModule->getLoc();
  // Unpack destination and source array and indices
  auto dest_array = operands[0];
  auto dest_idx = operands[1];
  auto src_array = operands[2];
  auto src_idx = operands[3];
  FlatSymbolRefAttr array_get_elem_fn_ptr = [&]() {
    static const std::string qir_get_qubit_from_array =
        "__quantum__rt__array_get_element_ptr_1d";
    if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_get_qubit_from_array)) {
      return SymbolRefAttr::get(qir_get_qubit_from_array, context);
    } else {
      // prototype should be (int64* : qreg, int64 : element) -> int64* :
      // qubit
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

      return mlir::SymbolRefAttr::get(qir_get_qubit_from_array, context);
    }
  }();

  // Create the CallOp for the get element ptr 1d function
  auto get_dest_qbit_qir_call = rewriter.create<mlir::CallOp>(
      location, array_get_elem_fn_ptr,
      LLVM::LLVMPointerType::get(IntegerType::get(context, 8)),
      llvm::makeArrayRef(std::vector<mlir::Value>{dest_array, dest_idx}));

  auto get_src_qbit_qir_call = rewriter.create<mlir::CallOp>(
      location, array_get_elem_fn_ptr,
      LLVM::LLVMPointerType::get(IntegerType::get(context, 8)),
      llvm::makeArrayRef(std::vector<mlir::Value>{src_array, src_idx}));

  // Load source qubit
  auto src_bitcast = rewriter.create<LLVM::BitcastOp>(
      location,
      LLVM::LLVMPointerType::get(
          LLVM::LLVMPointerType::get(get_quantum_type("Qubit", context))),
      get_src_qbit_qir_call.getResult(0));

  auto real_casted_src_qubit = rewriter.create<LLVM::LoadOp>(
      location, LLVM::LLVMPointerType::get(get_quantum_type("Qubit", context)),
      src_bitcast.res());

  // Destination: just cast the raw ptr to Qubit** to store the source Qubit*
  // to. Get the destination raw ptr (int8) and cast to Qubit**
  auto dest_bitcast = rewriter.create<LLVM::BitcastOp>(
      location,
      LLVM::LLVMPointerType::get(
          LLVM::LLVMPointerType::get(get_quantum_type("Qubit", context))),
      get_dest_qbit_qir_call.getResult(0));

  // Store source (Qubit*) to destination (Qubit**)
  rewriter.create<LLVM::StoreOp>(location, real_casted_src_qubit, dest_bitcast);
  rewriter.eraseOp(op);
  // std::cout << "After assign:\n";
  // parentModule.dump();
  return success();
}
} // namespace qcor