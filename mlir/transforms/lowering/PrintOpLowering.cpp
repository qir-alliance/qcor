/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "PrintOpLowering.hpp"

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

FlatSymbolRefAttr PrintOpLowering::getOrInsertPrintf(PatternRewriter &rewriter,
                                                     ModuleOp module) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
    return mlir::SymbolRefAttr::get("printf", context);

  // Create a function declaration for printf, the signature is:
  //   * `i32 (i8*, ...)`
  auto llvmI32Ty = IntegerType::get(context, 32);
  auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
  auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                /*isVarArg=*/true);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
  return mlir::SymbolRefAttr::get("printf", context);
}

/// Return a value representing an access into a global string with the given
/// name, creating the string if necessary.
Value PrintOpLowering::getOrCreateGlobalString(Location loc, OpBuilder &builder,
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
LogicalResult PrintOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // Local Declarations, get location, parentModule
  // and the context
  auto loc = op->getLoc();
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  auto context = parentModule->getContext();
  auto printOp = cast<mlir::quantum::PrintOp>(op);
  auto print_args = printOp.print_args();

  std::stringstream ss;

  std::string frmt_spec = "";
  std::size_t count = 0;
  std::vector<mlir::Value> args;
  for (auto operand : print_args) {
    if (operand.getType().isa<mlir::IntegerType>() ||
        operand.getType().isa<mlir::IndexType>()) {
      frmt_spec += "%d";
      ss << "_int_d_";
    } else if (operand.getType().isa<mlir::FloatType>()) {
      frmt_spec += "%lf";
      ss << "_float_f_";
    } else if (operand.getType().isa<mlir::OpaqueType>() &&
               operand.getType().cast<mlir::OpaqueType>().getTypeData() ==
                   "StringType") {
      frmt_spec += "%s";
      ss << "_string_s_";
    } else if (auto mem_ref_type =
                   operand.getType().dyn_cast_or_null<mlir::MemRefType>()) {
      if (mem_ref_type.getElementType().isa<mlir::IntegerType>() &&
          mem_ref_type.getRank() > 0 &&
          mem_ref_type.getElementType().getIntOrFloatBitWidth() == 1) {
        // This is a bit array...
        auto dim = mem_ref_type.getShape()[0];
        for (int i = 0; i < dim; i++) {
          frmt_spec += "%d";
        }
        ss << "_bit_array_b_" << dim;
      } else if (mem_ref_type.getElementType().isa<mlir::IntegerType>() &&
                 mem_ref_type.getRank() == 0 &&
                 mem_ref_type.getElementType().getIntOrFloatBitWidth() == 1) {
        frmt_spec += "%d";
        ss << "_bit_array_b_0";
      }
    } else {
      std::cout << "Currently invalid type to print.\n";
      operand.getType().dump();
      return failure();
    }
    count++;
    if (count < print_args.size()) {
      frmt_spec += " ";
    }
  }

  frmt_spec += "\n";

  auto printfRef = getOrInsertPrintf(rewriter, parentModule);
  Value formatSpecifierCst = getOrCreateGlobalString(
      loc, rewriter, "frmt_spec__" + ss.str(),
      StringRef(frmt_spec.c_str(), frmt_spec.length() + 1), parentModule);

  args.push_back(formatSpecifierCst);
  for (auto operand : print_args) {
    auto o = operand;
    if (o.getType().isa<mlir::FloatType>()) {
      // To display with printf, have to map to double with fpext
      auto type = mlir::FloatType::getF64(context);
      o = rewriter
              .create<LLVM::FPExtOp>(
                  loc, type, llvm::makeArrayRef(std::vector<mlir::Value>{o}))
              .res();
    } else if (o.getType().isa<mlir::OpaqueType>() &&
               operand.getType().cast<mlir::OpaqueType>().getTypeData() ==
                   "StringType") {
      auto op = o.getDefiningOp<mlir::quantum::CreateStringLiteralOp>();
      auto var_name = op.varname().str();
      o = variables[var_name];
    } else if (auto mem_ref_type =
                   o.getType().dyn_cast_or_null<mlir::MemRefType>()) {
      if (mem_ref_type.getElementType().isa<mlir::IntegerType>() &&
          mem_ref_type.getRank() > 0 &&
          mem_ref_type.getElementType().getIntOrFloatBitWidth() == 1) {
        // This is a bit array...

        auto dim = mem_ref_type.getShape()[0];
        for (int i = 0; i < dim; i++) {
          auto attr = mlir::IntegerAttr::get(rewriter.getIndexType(), i);
          auto ii = rewriter.create<mlir::ConstantOp>(loc, attr);
          auto z = rewriter.create<mlir::LoadOp>(
              loc, o, llvm::makeArrayRef(std::vector<mlir::Value>{ii}));
          args.push_back(z);
        }

        continue;
      } else if (mem_ref_type.getElementType().isa<mlir::IntegerType>() &&
                 mem_ref_type.getRank() == 0 &&
                 mem_ref_type.getElementType().getIntOrFloatBitWidth() == 1) {
        auto z = rewriter.create<mlir::LoadOp>(loc, o);
        args.push_back(z);
        continue;
      }
    }

    args.push_back(o);
  }
  rewriter.create<mlir::CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                                llvm::makeArrayRef(args));
  rewriter.eraseOp(op);

  // parentModule.dump();
  return success();
}
}  // namespace qcor