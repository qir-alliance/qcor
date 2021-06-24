#include "InstOpLowering.hpp"

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
// Match and replace all InstOps
LogicalResult InstOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // Local Declarations
  auto loc = op->getLoc();
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  auto context = parentModule->getContext();

  // Now get Instruction name and its quantum runtime function name
  auto instOp = cast<mlir::quantum::InstOp>(op);
  auto inst_name = instOp.name().str();
  inst_name = (inst_map.count(inst_name) ? inst_map[inst_name] : inst_name);

  // If this is a function we created, then we should convert the instop
  // to a llvm call op on that function
  // // Need to find the quantum instruction function
  // // Should be void __quantum__qis__INST(Qubit q) for example
  FlatSymbolRefAttr q_symbol_ref;
  std::string q_function_name = "__quantum__qis__" + inst_name;
  if (std::find(module_function_names.begin(), module_function_names.end(),
                llvm::StringRef(inst_name)) != module_function_names.end()) {
    q_function_name = inst_name;
  }

  // First see if this is a function within the mlir quantum dialect
  // then see if we've created this as an llvm function already,
  // finally, just create it as an llvm function
  if (parentModule.lookupSymbol<mlir::FuncOp>(q_function_name)) {
    q_symbol_ref = SymbolRefAttr::get(q_function_name, context);
  } else if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(q_function_name)) {
    q_symbol_ref = SymbolRefAttr::get(q_function_name, context);
  } else {
    // Return type should be void except for mz, which should be int64
    mlir::Type ret_type = LLVM::LLVMVoidType::get(context);
    if (inst_name == "mz") {
      ret_type =
          LLVM::LLVMPointerType::get(get_quantum_type("Result", context));
      // ret_type = rewriter.getIntegerType(1);
      // LLVM::LLVMPointerType::get(get_quantum_type("Result", context));
    }

    // Create Types for all function arguments, start with
    // double parameters (if instOp has them)
    std::vector<Type> tmp_arg_types;
    for (std::size_t i = 0; i < instOp.params().size(); i++) {
      auto param_type = FloatType::getF64(context);
      tmp_arg_types.push_back(param_type);
    }

    // Now, we need a QubitType for each qubit argument
    for (std::size_t i = 0; i < instOp.qubits().size(); i++) {
      // for (auto qbit : instOp.qubits()) {
      auto qubit_index_type =
          LLVM::LLVMPointerType::get(get_quantum_type("Qubit", context));
      tmp_arg_types.push_back(qubit_index_type);
    }

    // Create the LLVM FunctionType
    auto get_ptr_qbit_ftype = LLVM::LLVMFunctionType::get(
        ret_type, llvm::makeArrayRef(tmp_arg_types), false);

    // Insert the function since it hasn't been seen yet
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(parentModule.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(parentModule->getLoc(), q_function_name,
                                      get_ptr_qbit_ftype);
    q_symbol_ref = mlir::SymbolRefAttr::get(q_function_name, context);
  }

  // Now create the vector containing the function Values,
  // double parameters first if we have them...
  std::vector<mlir::Value> func_args;

  auto n_qbits = instOp.qubits().size();
  // Add the parameters first
  for (size_t i = n_qbits; i < operands.size(); i++) {
    func_args.push_back(operands[i]);
  }
  for (size_t i = 0; i < n_qbits; i++) {
    if (inst_name == "mz") {
      // Handle situation like
      // %4 = qvs.h(%1) : !quantum.Qubit
      // %5 = q.mz(%4) : (!quantum.Qubit) -> i1
      // 6 = index_cast %c0_i64 : i64 to index
      // store %5, %3[%6] : memref<i1>
      // %7 = qvs.reset(%4) : !quantum.Qubit
      // with 2 users of %4
      if (auto q_op =
              operands[i]
                  .getDefiningOp<mlir::quantum::ValueSemanticsInstOp>()) {
        func_args.push_back(q_op.getOperands()[0]);
        break;
      }
    }
    func_args.push_back(operands[i]);
  }

  // once again, return type should be void unless its a measure
  mlir::Type ret_type = LLVM::LLVMVoidType::get(context);
  if (inst_name == "mz") {
    ret_type =  // rewriter.getIntegerType(1);
        LLVM::LLVMPointerType::get(get_quantum_type("Result", context));
  }

  // Create the CallOp for this quantum instruction
  auto c = rewriter.create<mlir::CallOp>(loc, q_symbol_ref, ret_type,
                                         llvm::makeArrayRef(func_args));

  if (inst_name == "mz") {
    rewriter.replaceOp(op, c.getResult(0));
  } else {
    rewriter.eraseOp(op);
  }

  // Notify the rewriter that this operation has been removed.
  // rewriter.eraseOp(op);

  return success();
}

LogicalResult ResultCastOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op->getLoc();
  auto resultCastOp = cast<mlir::quantum::ResultCastOp>(op);
  auto qir_result = resultCastOp.measure_result();
  // Cast Result* -> Bool* (i1*)
  auto bitcast = rewriter.create<LLVM::BitcastOp>(
      loc, LLVM::LLVMPointerType::get(rewriter.getIntegerType(1)), qir_result);
  // Load bool from bool*
  auto bool_result = rewriter.create<LLVM::LoadOp>(
      loc, rewriter.getIntegerType(1), bitcast.res());
  rewriter.replaceOp(op, bool_result.res());

  return success();
}
}  // namespace qcor