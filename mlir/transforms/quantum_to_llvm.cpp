#include "quantum_to_llvm.hpp"

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
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
namespace {
using namespace mlir;
std::map<std::string, std::string> inst_map{{"cx", "cnot"}, {"measure", "mz"}};

class QallocOpLowering : public ConversionPattern {
 protected:
  std::string qir_qubit_array_allocate = "__quantum__rt__qubit_allocate_array";
  std::map<std::string, mlir::Value> &variables;

 public:
  explicit QallocOpLowering(MLIRContext *context,
                            std::map<std::string, mlir::Value> &vars)
      : ConversionPattern(mlir::quantum::QallocOp::getOperationName(), 1,
                          context),
        variables(vars) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();

    FlatSymbolRefAttr symbol_ref;
    if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_qubit_array_allocate)) {
      symbol_ref = SymbolRefAttr::get(qir_qubit_array_allocate, context);
    } else {
      auto qubit_type = LLVM::LLVMType::getInt64Ty(context);
      auto array_qbit_type = LLVM::LLVMType::getInt64Ty(context).getPointerTo();
      auto qalloc_ftype =
          LLVM::LLVMType::getFunctionTy(array_qbit_type, qubit_type, true);

      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(parentModule->getLoc(),
                                        qir_qubit_array_allocate, qalloc_ftype);

      symbol_ref = mlir::SymbolRefAttr::get(qir_qubit_array_allocate, context);
    }
    auto qallocOp = cast<mlir::quantum::QallocOp>(op);
    auto size = qallocOp.size();
    auto qreg_name = qallocOp.name().str();

    Value create_size_int = rewriter.create<LLVM::ConstantOp>(
        loc, LLVM::LLVMType::getInt64Ty(rewriter.getContext()),
        rewriter.getIntegerAttr(rewriter.getI64Type(), size));

    auto array_qbit_type = LLVM::LLVMType::getInt64Ty(context).getPointerTo();
    auto qalloc_qir_call = rewriter.create<mlir::CallOp>(
        loc, symbol_ref, array_qbit_type, ArrayRef<Value>({create_size_int}));

    auto qbit_array = qalloc_qir_call.getResult(0);

    rewriter.eraseOp(op);

    variables.insert({qreg_name, qbit_array});

    return success();
  }
};

class InstOpLowering : public ConversionPattern {
 protected:
  std::string qir_get_qubit_from_array =
      "__quantum__rt__array_get_element_ptr_1d";
  std::map<std::string, mlir::Value> &variables;
  std::map<mlir::Operation *, std::string> &qubit_extract_map;

 public:
  explicit InstOpLowering(MLIRContext *context,
                          std::map<std::string, mlir::Value> &vars,
                          std::map<mlir::Operation *, std::string> &qem)
      : ConversionPattern(mlir::quantum::InstOp::getOperationName(), 1,
                          context),
        variables(vars),
        qubit_extract_map(qem) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();

    // Now get Instruction name and the bits it operates on with qreg names
    auto instOp = cast<mlir::quantum::InstOp>(op);
    auto inst_name = instOp.name().str();
    inst_name = (inst_map.count(inst_name) ? inst_map[inst_name] : inst_name);

    std::vector<mlir::Value> qbit_results;
    for (auto operand : operands) {
      auto extract_op =
          operand.getDefiningOp<vector::ExtractElementOp>().getOperation();
      std::string get_qbit_call_qreg_key = qubit_extract_map[extract_op];
      mlir::Value qbit_result = variables[get_qbit_call_qreg_key];
      qbit_results.push_back(qbit_result);
    }

    // // Need to find the quantum instruction function
    // // Should be void __quantum__qis__INST(Qubit q) for example
    FlatSymbolRefAttr q_symbol_ref;
    std::string q_function_name =
        "__quantum__qis__" +
        (inst_map.count(inst_name) ? inst_map[inst_name] : inst_name);
    if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(q_function_name)) {
      q_symbol_ref = SymbolRefAttr::get(q_function_name, context);
    } else {
      LLVM::LLVMType ret_type = LLVM::LLVMType::getVoidTy(context);
      if (inst_name == "mz") {
        ret_type = LLVM::LLVMType::getInt64Ty(context);
      }

      std::vector<LLVM::LLVMType> tmp_arg_types;

      // FIXME loop over params too to add double types
      if (instOp.params()) {
        auto params = instOp.params().getValue();
        for (int i = 0; i < params.size(); i++) {
          auto param_type = LLVM::LLVMType::getDoubleTy(context);
          tmp_arg_types.push_back(param_type);
        }
      }

      // Need a Int64Type for each qubit argument
      for (int i = 0; i < operands.size(); i++) {
        auto qubit_index_type =
            LLVM::LLVMType::getInt64Ty(context).getPointerTo();
        tmp_arg_types.push_back(qubit_index_type);
      }

      // Create void (int, int) or void (int)
      auto get_ptr_qbit_ftype = LLVM::LLVMType::getFunctionTy(
          ret_type, llvm::makeArrayRef(tmp_arg_types), true);

      // Insert the function since it hasn't been seen yet
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(parentModule->getLoc(), q_function_name,
                                        get_ptr_qbit_ftype);

      q_symbol_ref = mlir::SymbolRefAttr::get(q_function_name, context);
    }

    std::vector<mlir::Value> func_args;
    if (instOp.params()) {
      auto params = instOp.params().getValue();
      for (std::uint64_t i = 0; i < params.getNumElements(); i++) {
        auto param_double =
            params.template getValue<double>(llvm::makeArrayRef({i}));
        std::cout << "HELLO inst_name: " << inst_name << ", " << param_double
                  << "\n";
        auto double_attr =
            mlir::FloatAttr::get(rewriter.getF64Type(), param_double);

        Value const_double_op = rewriter.create<LLVM::ConstantOp>(
            loc, LLVM::LLVMType::getDoubleTy(rewriter.getContext()),
            double_attr);

        func_args.push_back(const_double_op);
      }
    }

    for (auto q : qbit_results) {
      func_args.push_back(q);
    }

    LLVM::LLVMType ret_type = LLVM::LLVMType::getVoidTy(context);
    if (inst_name == "mz") {
      ret_type = LLVM::LLVMType::getInt64Ty(context);
    }

    auto qinst_qir_call = rewriter.create<mlir::CallOp>(
        loc, q_symbol_ref, ret_type, llvm::makeArrayRef(func_args));

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);

    return success();
  }
};

class ExtractQubitOpConversion : public ConversionPattern {
 protected:
  LLVMTypeConverter &typeConverter;
  std::map<std::string, mlir::Value> &vars;
  std::map<mlir::Operation *, std::string> &qubit_extract_map;

 public:
  explicit ExtractQubitOpConversion(
      MLIRContext *context, LLVMTypeConverter &c,
      std::map<std::string, mlir::Value> &v,
      std::map<mlir::Operation *, std::string> &qem)
      : ConversionPattern(mlir::vector::ExtractElementOp::getOperationName(), 1,
                          context),
        typeConverter(c),
        vars(v),
        qubit_extract_map(qem) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    auto adaptor = vector::ExtractElementOpAdaptor(operands);

    auto vectorType = cast<vector::ExtractElementOp>(op).getVectorType();

    auto llvmType = typeConverter.convertType(vectorType.getElementType());

    // LLVM::LLVMType::getInt64Ty(context).getPointerTo();

    // Bail if result type cannot be lowered.
    if (!llvmType) {
      return failure();
    }

    mlir::Value v = operands[0];
    mlir::Value v1 = operands[1];

    auto qalloc_op = v.getDefiningOp<quantum::QallocOp>();
    auto qbit_constant_op = v1.getDefiningOp<LLVM::ConstantOp>();

    // Get info about what qreg we are extracting what qbit from
    std::string qreg_name = qalloc_op.name().str();
    mlir::Attribute unknown_attr = qbit_constant_op.value();
    auto int_attr = unknown_attr.cast<mlir::IntegerAttr>();
    auto int_value = int_attr.getInt();
    auto qubit_var_name = qreg_name + "_" + std::to_string(int_value);

    // Erase the old op
    rewriter.eraseOp(op);

    // Reuse the qubit if we've allocated it before.
    if (vars.count(qubit_var_name)) {
      qubit_extract_map.insert(
          {op, qreg_name + "_" + std::to_string(int_value)});
      return success();
    }

    auto context = parentModule->getContext();
    std::string qir_get_qubit_from_array =
        "__quantum__rt__array_get_element_ptr_1d";
    // First goal, get symbol for __quantum__rt__array_get_element_ptr_1d
    // function
    FlatSymbolRefAttr symbol_ref;
    if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_get_qubit_from_array)) {
      symbol_ref = SymbolRefAttr::get(qir_get_qubit_from_array, context);
    } else {
      auto qubit_array_type =
          LLVM::LLVMType::getInt64Ty(context).getPointerTo();
      auto qubit_index_type = LLVM::LLVMType::getInt64Ty(context);

      auto qbit_element_ptr_type =
          LLVM::LLVMType::getInt64Ty(context).getPointerTo();
      auto get_ptr_qbit_ftype = LLVM::LLVMType::getFunctionTy(
          qbit_element_ptr_type,
          llvm::ArrayRef<LLVM::LLVMType>{qubit_array_type, qubit_index_type},
          true);

      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(
          parentModule->getLoc(), qir_get_qubit_from_array, get_ptr_qbit_ftype);

      symbol_ref = mlir::SymbolRefAttr::get(qir_get_qubit_from_array, context);
    }

    // Create the CallOp for the get element ptr 1d function
    auto array_qbit_type = LLVM::LLVMType::getInt64Ty(context).getPointerTo();
    auto get_qbit_qir_call = rewriter.create<mlir::CallOp>(
        parentModule->getLoc(), symbol_ref, array_qbit_type,
        ArrayRef<Value>({vars[qreg_name], adaptor.position()}));

    // Remember the variable name for this qubit
    vars.insert({qreg_name + "_" + std::to_string(int_value),
                 get_qbit_qir_call.getResult(0)});

    // STORE THAT THIS OP PRODUCES THIS QREG{IDX} VARIABLE NAME
    qubit_extract_map.insert({op, qreg_name + "_" + std::to_string(int_value)});

    return success();
  }
};

}  // namespace
namespace qcor {
void QuantumToLLVMLoweringPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}
void QuantumToLLVMLoweringPass::runOnOperation() {
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  LLVMTypeConverter typeConverter(&getContext());

  OwningRewritePatternList patterns;
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  // Common variables to share across converteres
  std::map<std::string, mlir::Value> variables;
  std::map<mlir::Operation *, std::string> qubit_extract_map;

  // Add our custom conversion passes
  patterns.insert<QallocOpLowering>(&getContext(), variables);
  patterns.insert<InstOpLowering>(&getContext(), variables, qubit_extract_map);
  patterns.insert<ExtractQubitOpConversion>(&getContext(), typeConverter,
                                            variables, qubit_extract_map);

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
}  // namespace qcor