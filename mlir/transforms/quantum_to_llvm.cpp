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
#include "quantum_dialect.hpp"

namespace {
using namespace mlir;
std::map<std::string, std::string> inst_map{{"cx", "cnot"}, {"measure", "mz"}};

mlir::Type get_quantum_type(std::string type,
                                      mlir::MLIRContext *context) {
  return LLVM::LLVMStructType::getOpaque(type, context);
}

// The goal of QallocOpLowering is to lower all occurrences of the
// MLIR QuantumDialect QallocOp to the MSFT QIR
// __quantum__rt__qubit_allocate_array() quantum runtime function as an LLVM
// MLIR Function and CallOp.
class QallocOpLowering : public ConversionPattern {
 protected:
  // Constant string for runtime function name
  inline static const std::string qir_qubit_array_allocate =
      "__quantum__rt__qubit_allocate_array";
  // Rudimentary symbol table, seen variables
  std::map<std::string, mlir::Value> &variables;

  // %Array* @__quantum__rt__qubit_allocate_array(i64 %nQubits)
 public:
  // Constructor, store seen variables
  explicit QallocOpLowering(MLIRContext *context,
                            std::map<std::string, mlir::Value> &vars)
      : ConversionPattern(mlir::quantum::QallocOp::getOperationName(), 1,
                          context),
        variables(vars) {}

  // Match any Operation that is the QallocOp
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Local Declarations, get location, parentModule
    // and the context
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();

    // First step is to get a reference to the Symbol Reference for the
    // qalloc QIR runtime function, this will only declare it once and reuse
    // each time it is seen
    FlatSymbolRefAttr symbol_ref;
    if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_qubit_array_allocate)) {
      symbol_ref = SymbolRefAttr::get(qir_qubit_array_allocate, context);
    } else {
      // prototype is (size : int64) -> Array* : qubit_array_ptr
      auto qubit_type = IntegerType::get(context, 64);
      auto array_qbit_type =
          LLVM::LLVMPointerType::get(get_quantum_type("Array", context));
      auto qalloc_ftype =
          LLVM::LLVMFunctionType::get(array_qbit_type, qubit_type, false);

      // Insert the function declaration
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(parentModule->getLoc(),
                                        qir_qubit_array_allocate, qalloc_ftype);
      symbol_ref = mlir::SymbolRefAttr::get(qir_qubit_array_allocate, context);
    }

    // Get as a QallocOp, get its allocatino size and qreg variable name
    auto qallocOp = cast<mlir::quantum::QallocOp>(op);
    auto size = qallocOp.size();
    auto qreg_name = qallocOp.name().str();

    // create a CallOp for the new quantum runtime allocation
    // function.
    // size_value = constantop (size)
    // qubit_array_ptr = callop ( size_value )
    Value create_size_int = rewriter.create<LLVM::ConstantOp>(
        loc, IntegerType::get(rewriter.getContext(), 64),
        rewriter.getIntegerAttr(rewriter.getI64Type(), size));
    auto array_qbit_type =
        LLVM::LLVMPointerType::get(get_quantum_type("Array", context));
    auto qalloc_qir_call = rewriter.create<mlir::CallOp>(
        loc, symbol_ref, array_qbit_type, ArrayRef<Value>({create_size_int}));

    // Get the returned qubit array pointer Value
    auto qbit_array = qalloc_qir_call.getResult(0);

    // Remove the old QuantumDialect QallocOp
    rewriter.eraseOp(op);

    // Save the qubit array variable to the symbol table
    variables.insert({qreg_name, qbit_array});

    return success();
  }
};

// declare void @__quantum__rt__qubit_release_array(%Array*)
class DeallocOpLowering : public ConversionPattern {
 protected:
  // Constant string for runtime function name
  inline static const std::string qir_qubit_array_deallocate =
      "__quantum__rt__qubit_release_array";
  // Rudimentary symbol table, seen variables
  std::map<std::string, mlir::Value> &variables;

  // %Array* @__quantum__rt__qubit_allocate_array(i64 %nQubits)
 public:
  // Constructor, store seen variables
  explicit DeallocOpLowering(MLIRContext *context,
                             std::map<std::string, mlir::Value> &vars)
      : ConversionPattern(mlir::quantum::DeallocOp::getOperationName(), 1,
                          context),
        variables(vars) {}

  // Match any Operation that is the QallocOp
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Local Declarations, get location, parentModule
    // and the context
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();

    // First step is to get a reference to the Symbol Reference for the
    // qalloc QIR runtime function, this will only declare it once and reuse
    // each time it is seen
    FlatSymbolRefAttr symbol_ref;
    if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(
            qir_qubit_array_deallocate)) {
      symbol_ref = SymbolRefAttr::get(qir_qubit_array_deallocate, context);
    } else {
      // prototype is (Array*) -> void
      auto void_type = LLVM::LLVMVoidType::get(context);
      auto array_qbit_type =
          LLVM::LLVMPointerType::get(get_quantum_type("Array", context));
      auto dealloc_ftype =
          LLVM::LLVMFunctionType::get(void_type, array_qbit_type, false);

      // Insert the function declaration
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(
          parentModule->getLoc(), qir_qubit_array_deallocate, dealloc_ftype);
      symbol_ref =
          mlir::SymbolRefAttr::get(qir_qubit_array_deallocate, context);
    }

    // Get as a QallocOp, get its allocatino size and qreg variable name
    auto deallocOp = cast<mlir::quantum::DeallocOp>(op);
    auto qubits_value = deallocOp.qubits();
    auto qreg_name_attr = qubits_value.getDefiningOp()->getAttr("name");
    auto name = qreg_name_attr.cast<::mlir::StringAttr>().getValue();
    auto qubits = variables[name.str()];

    // create a CallOp for the new quantum runtime de-allocation
    // function.
    rewriter.create<mlir::CallOp>(loc, symbol_ref,
                                  LLVM::LLVMVoidType::get(context),
                                  ArrayRef<Value>({qubits}));

    // Remove the old QuantumDialect QallocOp
    rewriter.eraseOp(op);

    return success();
  }
};

class QRTInitOpLowering : public ConversionPattern {
 protected:
  // Constant string for runtime function name
  inline static const std::string qir_qrt_initialize =
      "__quantum__rt__initialize";
  // Rudimentary symbol table, seen variables
  std::map<std::string, mlir::Value> &variables;

  // %Array* @__quantum__rt__qubit_allocate_array(i64 %nQubits)
 public:
  // Constructor, store seen variables
  explicit QRTInitOpLowering(MLIRContext *context,
                             std::map<std::string, mlir::Value> &vars)
      : ConversionPattern(mlir::quantum::QRTInitOp::getOperationName(), 1,
                          context),
        variables(vars) {}

  // Match any Operation that is the QallocOp
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Local Declarations, get location, parentModule
    // and the context
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();

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

    // auto initOp = cast<mlir::quantum::QRTInitOp>(op);
    // auto parentFunc = op->getParentOfType<LLVM::LLVMFuncOp>();
    // parentFunc.dump();
    // auto args = parentFunc.body().getArguments();
    // std::cout << "HERE:\n";
    // args[0].dump();
    // args[1].dump();
    // create a CallOp for the new quantum runtime initialize
    // function.
    rewriter.create<mlir::CallOp>(
        loc, symbol_ref, IntegerType::get(context, 32),
        ArrayRef<Value>({variables["main_argc"], variables["main_argv"]}));

    // Remove the old QuantumDialect QallocOp
    rewriter.eraseOp(op);

    return success();
  }
};

class QRTFinalizeOpLowering : public ConversionPattern {
 protected:
  // Constant string for runtime function name
  inline static const std::string qir_qrt_finalize = "__quantum__rt__finalize";
  // Rudimentary symbol table, seen variables
  std::map<std::string, mlir::Value> &variables;

  // %Array* @__quantum__rt__qubit_allocate_array(i64 %nQubits)
 public:
  // Constructor, store seen variables
  explicit QRTFinalizeOpLowering(MLIRContext *context,
                                 std::map<std::string, mlir::Value> &vars)
      : ConversionPattern(mlir::quantum::QRTFinalizeOp::getOperationName(), 1,
                          context),
        variables(vars) {}

  // Match any Operation that is the QallocOp
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Local Declarations, get location, parentModule
    // and the context
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();

    // First step is to get a reference to the Symbol Reference for the
    // qalloc QIR runtime function, this will only declare it once and reuse
    // each time it is seen
    FlatSymbolRefAttr symbol_ref;
    if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_qrt_finalize)) {
      symbol_ref = SymbolRefAttr::get(qir_qrt_finalize, context);
    } else {
      // prototype is () -> void
      auto void_type = LLVM::LLVMVoidType::get(context);
      std::vector<mlir::Type> arg_types;
      auto init_ftype = LLVM::LLVMFunctionType::get(
          void_type, llvm::makeArrayRef(arg_types), false);

      // Insert the function declaration
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(parentModule->getLoc(),
                                        qir_qrt_finalize, init_ftype);
      symbol_ref = mlir::SymbolRefAttr::get(qir_qrt_finalize, context);
    }

    // auto initOp = cast<mlir::quantum::QRTInitOp>(op);
    // auto parentFunc = op->getParentOfType<LLVM::LLVMFuncOp>();
    // parentFunc.dump();
    // auto args = parentFunc.body().getArguments();
    // std::cout << "HERE:\n";
    // args[0].dump();
    // args[1].dump();
    // create a CallOp for the new quantum runtime initialize
    // function.
    rewriter.create<mlir::CallOp>(
        loc, symbol_ref, LLVM::LLVMVoidType::get(context), ArrayRef<Value>({}));

    // Remove the old QuantumDialect QallocOp
    rewriter.eraseOp(op);

    return success();
  }
};

class SetQregOpLowering : public ConversionPattern {
 protected:
  // Constant string for runtime function name
  inline static const std::string qir_qrt_finalize =
      "__quantum__rt__set_external_qreg";
  // Rudimentary symbol table, seen variables
  std::map<std::string, mlir::Value> &variables;

  // %Array* @__quantum__rt__qubit_allocate_array(i64 %nQubits)
 public:
  // Constructor, store seen variables
  explicit SetQregOpLowering(MLIRContext *context,
                             std::map<std::string, mlir::Value> &vars)
      : ConversionPattern(mlir::quantum::SetQregOp::getOperationName(), 1,
                          context),
        variables(vars) {}

  // Match any Operation that is the QallocOp
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Local Declarations, get location, parentModule
    // and the context
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();

    // First step is to get a reference to the Symbol Reference for the
    // qalloc QIR runtime function, this will only declare it once and reuse
    // each time it is seen
    FlatSymbolRefAttr symbol_ref;
    if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_qrt_finalize)) {
      symbol_ref = SymbolRefAttr::get(qir_qrt_finalize, context);
    } else {
      // prototype is () -> void
      auto void_type = LLVM::LLVMVoidType::get(context);
      std::vector<mlir::Type> arg_types{
          LLVM::LLVMPointerType::get(get_quantum_type("qreg", context))};
      auto init_ftype = LLVM::LLVMFunctionType::get(
          void_type, llvm::makeArrayRef(arg_types), false);

      // Insert the function declaration
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(parentModule->getLoc(),
                                        qir_qrt_finalize, init_ftype);
      symbol_ref = mlir::SymbolRefAttr::get(qir_qrt_finalize, context);
    }

    // auto initOp = cast<mlir::quantum::QRTInitOp>(op);
    // auto parentFunc = op->getParentOfType<LLVM::LLVMFuncOp>();
    // parentFunc.dump();
    // auto args = parentFunc.body().getArguments();
    // std::cout << "HERE:\n";
    // args[0].dump();
    // args[1].dump();
    // create a CallOp for the new quantum runtime initialize
    // function.
    rewriter.create<mlir::CallOp>(
        loc, symbol_ref, LLVM::LLVMVoidType::get(context),
        ArrayRef<Value>({variables["_incoming_qreg_variable"]}));

    // Remove the old QuantumDialect QallocOp
    rewriter.eraseOp(op);

    return success();
  }
};

class QuantumFuncArgConverter : public ConversionPattern {
 protected:
  std::unique_ptr<mlir::TypeConverter> my_tc;
  MLIRContext *context;
  std::map<std::string, mlir::Value> &variables;

 public:
  explicit QuantumFuncArgConverter(MLIRContext *ctx,
                                   std::map<std::string, mlir::Value> &vars)
      : ConversionPattern(mlir::FuncOp::getOperationName(), 1, ctx),
        context(ctx),
        variables(vars) {
    my_tc = std::make_unique<mlir::TypeConverter>();
    my_tc->addConversion([this](mlir::Type type) -> mlir::Optional<mlir::Type> {
      if (type.isa<mlir::OpaqueType>()) {
        auto casted = type.cast<mlir::OpaqueType>();
        if (casted.getTypeData() == "Qubit") {
          return LLVM::LLVMPointerType::get(
              get_quantum_type("Qubit", this->context));
        } else if (casted.getTypeData() == "ArgvType") {
          return LLVM::LLVMPointerType::get(
              LLVM::LLVMPointerType::get(IntegerType::get(context, 8)));
        } else if (casted.getTypeData() == "qreg") {
          return LLVM::LLVMPointerType::get(
              get_quantum_type("qreg", this->context));
        }
      } else if (type.isa<mlir::IntegerType>()) {
        return IntegerType::get(this->context, 32);
      }
      return llvm::None;
    });
    typeConverter = my_tc.get();
  }
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();
    auto funcOp = cast<mlir::FuncOp>(op);
    auto ftype = funcOp.type().cast<FunctionType>();

    auto func_name = funcOp.getName().str();

    if (func_name == "main") {
      auto charstarstar = LLVM::LLVMPointerType::get(
          LLVM::LLVMPointerType::get(IntegerType::get(context, 8)));
      std::vector<Type> tmp_arg_types{IntegerType::get(context, 32),
                                      charstarstar};

      auto new_main_signature =
          LLVM::LLVMFunctionType::get(IntegerType::get(context, 32),
                                      llvm::makeArrayRef(tmp_arg_types), false);

      auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(loc, funcOp.sym_name(),
                                                         new_main_signature);
      rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                  newFuncOp.end());
      if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(),
                                             *typeConverter))) {
        return failure();
      }
      // rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter);

      auto args = newFuncOp.body().getArguments();
      variables.insert({"main_argc", args[0]});
      variables.insert({"main_argv", args[1]});

      rewriter.eraseOp(op);
      return success();
    }

    if (ftype.getNumInputs() == 1 &&
        ftype.getInput(0).isa<mlir::OpaqueType>() &&
        ftype.getInput(0).cast<mlir::OpaqueType>().getTypeData() == "qreg") {
      std::vector<mlir::Type> tmp_arg_types{
          LLVM::LLVMPointerType::get(get_quantum_type("qreg", context))};

      auto new_func_signature =
          LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context),
                                 llvm::makeArrayRef(tmp_arg_types), false);

      auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(loc, funcOp.sym_name(),
                                                         new_func_signature);

      rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                  newFuncOp.end());

      if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(),
                                             *typeConverter))) {
        return failure();
      }

      rewriter.eraseOp(op);
      auto arg = newFuncOp.body().getArguments()[0];

      variables.insert({"_incoming_qreg_variable", arg});
      return success();
    }

    // Not main, sub quantum kernel, convert Qubit to Qubit*
    if (ftype.getNumInputs() > 0) {
      std::vector<mlir::Type> tmp_arg_types;
      for (unsigned i = 0; i < ftype.getNumInputs(); i++) {
        tmp_arg_types.push_back(
            LLVM::LLVMPointerType::get(get_quantum_type("Qubit", context)));
      }

      auto new_func_signature = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(context), llvm::makeArrayRef(tmp_arg_types), false);

      auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(loc, funcOp.sym_name(),
                                                         new_func_signature);

      rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                  newFuncOp.end());
      // rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter);
      if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(),
                                             *typeConverter))) {
        return failure();
      }
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

// The goal of InstOpLowering is to convert all QuantumDialect
// InstOp (quantum.inst) to the corresponding __quantum__qis__INST(int64*, ...)
// call
class InstOpLowering : public ConversionPattern {
 protected:
  // Symbol table, local seen variables
  std::map<std::string, mlir::Value> &variables;

  // Mapping of Vector::ExtractElementOp Operation pointers to the
  // corresponding qreg variable name they represent
  std::map<mlir::Operation *, std::string> &qubit_extract_map;

  std::vector<std::string> &module_function_names;

 public:
  // The Constructor, store the variables and qubit extract op map
  explicit InstOpLowering(MLIRContext *context,
                          std::map<std::string, mlir::Value> &vars,
                          std::map<mlir::Operation *, std::string> &qem,
                          std::vector<std::string> &f_names)
      : ConversionPattern(mlir::quantum::InstOp::getOperationName(), 1,
                          context),
        variables(vars),
        qubit_extract_map(qem),
        module_function_names(f_names) {}

  // Match and replace all InstOps
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
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

    // The Operands at this point can only be the
    // qubits the InstOp is operating on, so lets get tehm
    // as mlir::Values to be used in the creation of the CallOp for this
    // quantum runtime function
    std::vector<mlir::Value> qbit_results;
    for (auto operand : operands) {
      // The Operand points to the vector::ExtractElementOp that produces the
      // qubit Value, get that Operation

      auto extract_op =
          operand.getDefiningOp<quantum::ExtractQubitOp>().getOperation();
      if (!extract_op) {
        if (operand.isa<BlockArgument>()) {
          qbit_results.push_back(operand);
        } else {
          std::cout << "Failure creating LLVM CallOp qubit value for instop "
                    << inst_name << "\n";
          return mlir::failure();
        }
      } else {
        // Now get the corresponding qubit variable name (q_0 for q[0])
        std::string get_qbit_call_qreg_key = qubit_extract_map[extract_op];
        // Now get the qubit Value from the symbol table
        mlir::Value qbit_result = variables[get_qbit_call_qreg_key];

        // Store those values for the CallOp
        qbit_results.push_back(qbit_result);
      }
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
      }

      // Create Types for all function arguments, start with
      // double parameters (if instOp has them)
      std::vector<Type> tmp_arg_types;
      if (instOp.params()) {
        auto params = instOp.params().getValue();
        for (int i = 0; i < params.size(); i++) {
          auto param_type = FloatType::getF64(context);
          tmp_arg_types.push_back(param_type);
        }
      }

      // Now, we need a Int64Type for each qubit argument
      for (std::size_t i = 0; i < operands.size(); i++) {
        auto qubit_index_type =
            LLVM::LLVMPointerType::get(get_quantum_type("Qubit", context));
        // IntegerType::get(context, 64).getPointerTo();
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
    if (instOp.params()) {
      auto params = instOp.params().getValue();
      for (std::int64_t i = 0; i < params.size(); i++) {
        auto param_double = params.template getValue<double>(
            llvm::makeArrayRef({(std::uint64_t)i}));
        auto double_attr =
            mlir::FloatAttr::get(rewriter.getF64Type(), param_double);

        Value const_double_op = rewriter.create<LLVM::ConstantOp>(
            loc, FloatType::getF64(rewriter.getContext()), double_attr);

        func_args.push_back(const_double_op);
      }
    }

    // Followed by qubit values
    for (auto q : qbit_results) {
      func_args.push_back(q);
    }

    // once again, return type should be void unless its a measure
    mlir::Type ret_type = LLVM::LLVMVoidType::get(context);
    if (inst_name == "mz") {
      ret_type =
          LLVM::LLVMPointerType::get(get_quantum_type("Result", context));
    }

    // Create the CallOp for this quantum instruction
    rewriter.create<mlir::CallOp>(loc, q_symbol_ref, ret_type,
                                  llvm::makeArrayRef(func_args));

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);

    return success();
  }
};

// The goal of this OpConversion is to map vector.extract on a
// qalloc qubit vector to the MSFT QIR __quantum__rt__array_get_element_ptr_1d()
// call
class ExtractQubitOpConversion : public ConversionPattern {
 protected:
  LLVMTypeConverter &typeConverter;
  inline static const std::string qir_get_qubit_from_array =
      "__quantum__rt__array_get_element_ptr_1d";
  std::map<std::string, mlir::Value> &vars;
  std::map<mlir::Operation *, std::string> &qubit_extract_map;

 public:
  explicit ExtractQubitOpConversion(
      MLIRContext *context, LLVMTypeConverter &c,
      std::map<std::string, mlir::Value> &v,
      std::map<mlir::Operation *, std::string> &qem)
      : ConversionPattern(mlir::quantum::ExtractQubitOp::getOperationName(), 1,
                          context),
        typeConverter(c),
        vars(v),
        qubit_extract_map(qem) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Local Declarations
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();
    auto location = parentModule->getLoc();
    auto adaptor = quantum::ExtractQubitOpAdaptor(operands);

    // This Extract Op references the QallocOp it is operation on
    // and a constant op that represents the element to extract
    mlir::Value v = operands[0];
    mlir::Value v1 = operands[1];
    auto qalloc_op = v.getDefiningOp<quantum::QallocOp>();
    auto qbit_constant_op = v1.getDefiningOp<LLVM::ConstantOp>();

    // Get info about what qreg we are extracting what qbit from
    // Create the qubit variable name that we are extracting
    // e.g. q[0] -> q_0
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
        location, symbol_ref, array_qbit_type,
        ArrayRef<Value>({vars[qreg_name], adaptor.idx()}));

    auto bitcast = rewriter.create<LLVM::BitcastOp>(
        location,
        LLVM::LLVMPointerType::get(
            LLVM::LLVMPointerType::get(get_quantum_type("Qubit", context))),
        get_qbit_qir_call.getResult(0));
    auto real_casted_qubit = rewriter.create<LLVM::LoadOp>(
        location,
        LLVM::LLVMPointerType::get(get_quantum_type("Qubit", context)),
        bitcast.res());

    // Remember the variable name for this qubit
    vars.insert(
        {qreg_name + "_" + std::to_string(int_value), real_casted_qubit.res()});

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
  patterns.insert<QuantumFuncArgConverter>(&getContext(), variables);
  patterns.insert<QallocOpLowering>(&getContext(), variables);
  patterns.insert<InstOpLowering>(&getContext(), variables, qubit_extract_map,
                                  function_names);
  patterns.insert<SetQregOpLowering>(&getContext(), variables);
  patterns.insert<ExtractQubitOpConversion>(&getContext(), typeConverter,
                                            variables, qubit_extract_map);
  patterns.insert<DeallocOpLowering>(&getContext(), variables);
  patterns.insert<QRTInitOpLowering>(&getContext(), variables);
  patterns.insert<QRTFinalizeOpLowering>(&getContext(), variables);

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
}  // namespace qcor