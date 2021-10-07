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
#include "QRTInitOpLowering.hpp"

#include <iostream>

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
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
namespace qcor {

Value local_getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                    StringRef name, StringRef value,
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
LogicalResult QRTInitOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // Local Declarations, get location, parentModule
  // and the context
  auto loc = op->getLoc();
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  auto context = parentModule->getContext();
  auto init_op = cast<mlir::quantum::QRTInitOp>(op);

  // First step is to get a reference to the Symbol Reference for the
  // qalloc QIR runtime function, this will only declare it once and reuse
  // each time it is seen
  FlatSymbolRefAttr symbol_ref;
  if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_qrt_initialize)) {
    symbol_ref = SymbolRefAttr::get(qir_qrt_initialize, context);
  } else {
    // prototype is (Array*) -> void
    auto int_type = IntegerType::get(context, 32);
    std::vector<mlir::Type> arg_types{
        IntegerType::get(context, 32),
        LLVM::LLVMPointerType::get(
            LLVM::LLVMPointerType::get(IntegerType::get(context, 8)))};
    auto init_ftype = LLVM::LLVMFunctionType::get(
        int_type, llvm::makeArrayRef(arg_types), false);

    // Insert the function declaration
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(parentModule.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(parentModule->getLoc(),
                                      qir_qrt_initialize, init_ftype);
    symbol_ref = mlir::SymbolRefAttr::get(qir_qrt_initialize, context);
  }

  // Handle main() config attrs
  auto main_attrs = init_op.extra_argsAttr();
  if (!main_attrs.empty()) {
    FlatSymbolRefAttr config_symbol_ref;
    if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_qrt_set_config)) {
      config_symbol_ref = SymbolRefAttr::get(qir_qrt_set_config, context);
    } else {
      // prototype is (int8_t*, int8_t*) -> void
      auto int_ptr_type =
          LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
      std::vector<mlir::Type> arg_types{int_ptr_type, int_ptr_type};
      auto config_set_init_ftype =
          LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context),
                                      llvm::makeArrayRef(arg_types), false);

      // Insert the function declaration
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(
          parentModule->getLoc(), qir_qrt_set_config, config_set_init_ftype);
      config_symbol_ref = mlir::SymbolRefAttr::get(qir_qrt_set_config, context);
    }

    for (std::size_t i = 0; i < main_attrs.size()-1; i += 2) {
      auto key = main_attrs[i].cast<StringAttr>().getValue();//.str();
      auto value = main_attrs[i + 1].cast<StringAttr>().getValue();//.str();

      // // Store key:key
      auto key_global = local_getOrCreateGlobalString(
          loc, rewriter, key,
          StringRef(key.str().c_str(), key.str().length() + 1), parentModule);

      // StringRef(frmt_spec.c_str(), frmt_spec.length() + 1
      // Store key_value:value
      auto value_global = local_getOrCreateGlobalString(
          loc, rewriter, key.str() + "_value",
          // MUST DO IT THIS WAY. Need null-terminator in there. 
          StringRef(value.str().c_str(), value.str().length() + 1),
          parentModule);

      rewriter.create<mlir::CallOp>(loc, config_symbol_ref,
                                    LLVM::LLVMVoidType::get(context),
                                    llvm::makeArrayRef(std::vector<mlir::Value>{
                                        key_global, value_global}));
    }
  }

  // create a CallOp for the new quantum runtime initialize
  // function.
  rewriter.create<mlir::CallOp>(loc, symbol_ref, IntegerType::get(context, 32),
                                operands);

  // Remove the old QuantumDialect QallocOp
  rewriter.eraseOp(op);

  return success();
}
}  // namespace qcor