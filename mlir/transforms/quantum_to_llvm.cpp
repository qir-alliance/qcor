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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"

namespace {
using namespace mlir;
std::map<std::string, std::string> inst_map{{"cx", "cnot"}, {"measure", "mz"}};

mlir::Type get_quantum_type(std::string type, mlir::MLIRContext *context) {
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
    rewriter.replaceOp(op, qbit_array);
    rewriter.eraseOp(op);
    // Save the qubit array variable to the symbol table
    variables.insert({qreg_name, qbit_array});

    return success();
  }
};

// The goal of QubitArrayAllocOpLowering is to lower all occurrences of the
// MLIR QuantumDialect createQubitArray to the MSFT QIR
// __quantum__rt__array_create_1d() quantum runtime function for Qubit*
// (create a generic array holding references to Qubit for aliasing purposes)
// as an LLVM MLIR Function and CallOp.
class QubitArrayAllocOpLowering : public ConversionPattern {
protected:
  // Constant string for runtime function name
  inline static const std::string qir_qubit_array_allocate =
      "__quantum__rt__array_create_1d";
  // Rudimentary symbol table, seen variables
  std::map<std::string, mlir::Value> &variables;
  /// Lower to:
  /// %Array* @__quantum__rt__array_create_1d(i32 %elementSizeInBytes, i64% nQubits) 
  /// where elementSizeInBytes = 8 (pointer size).
public:
  // Constructor, store seen variables
  explicit QubitArrayAllocOpLowering(MLIRContext *context,
                                     std::map<std::string, mlir::Value> &vars)
      : ConversionPattern(mlir::quantum::QaliasArrayAllocOp::getOperationName(), 1,
                          context),
        variables(vars) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Local Declarations, get location, parentModule
    // and the context
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();

    // First step is to get a reference to the Symbol Reference for the
    // __quantum__rt__array_create_1d QIR runtime function,
    // this will only declare it once and reuse each time it is seen
    FlatSymbolRefAttr symbol_ref;
    if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_qubit_array_allocate)) {
      symbol_ref = SymbolRefAttr::get(qir_qubit_array_allocate, context);
    } else {
      // prototype is (elementSize: int32, arraySize : int64) -> Array* :
      // qubit_array_ptr
      auto qubit_type = IntegerType::get(context, 64);
      auto element_size_type = IntegerType::get(context, 32);
      auto array_qbit_type =
          LLVM::LLVMPointerType::get(get_quantum_type("Array", context));
      auto array_alloc_ftype = LLVM::LLVMFunctionType::get(
          array_qbit_type, llvm::ArrayRef<Type>{element_size_type, qubit_type},
          false);

      // Insert the function declaration
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(
          parentModule->getLoc(), qir_qubit_array_allocate, array_alloc_ftype);
      symbol_ref = mlir::SymbolRefAttr::get(qir_qubit_array_allocate, context);
    }

    // Get as a QaliasArrayAllocOp, get its allocation size and qreg variable
    // name
    auto qallocOp = cast<mlir::quantum::QaliasArrayAllocOp>(op);
    auto size = qallocOp.size();
    auto qreg_name = qallocOp.name().str();

    Value create_size_int = rewriter.create<LLVM::ConstantOp>(
        loc, IntegerType::get(rewriter.getContext(), 64),
        rewriter.getIntegerAttr(rewriter.getI64Type(), size));

    Value element_size_int = rewriter.create<LLVM::ConstantOp>(
        loc, IntegerType::get(rewriter.getContext(), 32),
        rewriter.getIntegerAttr(
            rewriter.getI64Type(),
            /* element size = pointer size */ sizeof(void *)));

    auto array_qbit_type =
        LLVM::LLVMPointerType::get(get_quantum_type("Array", context));
    auto qalloc_qir_call = rewriter.create<mlir::CallOp>(
        loc, symbol_ref, array_qbit_type,
        ArrayRef<Value>({element_size_int, create_size_int}));

    // Get the returned qubit array pointer Value
    auto qbit_array = qalloc_qir_call.getResult(0);

    // Remove the old QuantumDialect QallocOp
    rewriter.replaceOp(op, qbit_array);
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

    // create a CallOp for the new quantum runtime initialize
    // function.
    rewriter.create<mlir::CallOp>(
        loc, symbol_ref, IntegerType::get(context, 32), operands);

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

    // create a CallOp for the new quantum runtime initialize
    // function.
    rewriter.create<mlir::CallOp>(
        loc, symbol_ref, LLVM::LLVMVoidType::get(context), operands);

    // Remove the old QuantumDialect QallocOp
    rewriter.eraseOp(op);

    return success();
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

    auto n_params = instOp.params().size();
    auto n_qbits = instOp.qubits().size();
    for (int i = n_params + n_qbits - 1; i >= 0; i--) {
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
      auto bitcast = rewriter.create<LLVM::BitcastOp>(
          loc, LLVM::LLVMPointerType::get(rewriter.getIntegerType(1)),
          c.getResult(0));
      auto o = rewriter.create<LLVM::LoadOp>(loc, rewriter.getIntegerType(1),
                                             bitcast.res());
      rewriter.replaceOp(op, o.res());
    } else {
      rewriter.eraseOp(op);
    }

    // Notify the rewriter that this operation has been removed.
    // rewriter.eraseOp(op);

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
        location, symbol_ref, array_qbit_type, operands);
    // ArrayRef<Value>({vars[qreg_name], adaptor.idx()}));

    auto bitcast = rewriter.create<LLVM::BitcastOp>(
        location,
        LLVM::LLVMPointerType::get(
            LLVM::LLVMPointerType::get(get_quantum_type("Qubit", context))),
        get_qbit_qir_call.getResult(0));
    auto real_casted_qubit = rewriter.create<LLVM::LoadOp>(
        location,
        LLVM::LLVMPointerType::get(get_quantum_type("Qubit", context)),
        bitcast.res());

    rewriter.replaceOp(op, real_casted_qubit.res());
    // Remember the variable name for this qubit
    // vars.insert({qubit_var_name, real_casted_qubit.res()});

    // STORE THAT THIS OP PRODUCES THIS QREG{IDX} VARIABLE NAME
    // qubit_extract_map.insert({op, qubit_var_name});

    return success();
  }
};

class AssignQubitOpConversion : public ConversionPattern {
protected:
  std::map<std::string, mlir::Value> &variables;

public:
  // CTor: store seen variables
  explicit AssignQubitOpConversion(MLIRContext *context,
                                   std::map<std::string, mlir::Value> &vars)
      : ConversionPattern(
            mlir::quantum::AssignQubitOp::getOperationName(), 1,
            context),
        variables(vars) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Local Declarations
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();
    auto location = parentModule->getLoc();
    // Source and Destinations are Qubit* type
    auto dest = operands[0];
    auto src = operands[1];
    // Cast source pointer to Qubit**
    auto bitcast = rewriter.create<LLVM::BitcastOp>(
        location,
        LLVM::LLVMPointerType::get(
            LLVM::LLVMPointerType::get(get_quantum_type("Qubit", context))),
        src);
    // Store source (Qubit**) to destination
    // auto store_qubit_ptr =
    //     rewriter.create<LLVM::StoreOp>(location, bitcast.res(), dest);

    return success();
  }
};

class CreateStringLiteralOpLowering : public ConversionPattern {
 private:
  std::map<std::string, mlir::Value> &variables;

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
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

 public:
  // Constructor, store seen variables
  explicit CreateStringLiteralOpLowering(MLIRContext *context,
                                         std::map<std::string, mlir::Value> &v)
      : ConversionPattern(
            mlir::quantum::CreateStringLiteralOp::getOperationName(), 1,
            context),
        variables(v) {}

  // Match any Operation that is the QallocOp
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Local Declarations, get location, parentModule
    // and the context
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto slOp = cast<mlir::quantum::CreateStringLiteralOp>(op);
    auto slOpText = slOp.text();
    auto slVarName = slOp.varname();

    Value new_global_str = getOrCreateGlobalString(
        loc, rewriter, slVarName,
        StringRef(slOpText.str().c_str(), slOpText.str().length() + 1),
        parentModule);

    variables.insert({slVarName.str(), new_global_str});

    rewriter.eraseOp(op);

    return success();
  }
};

class PrintOpLowering : public ConversionPattern {
 private:
  std::map<std::string, mlir::Value> &variables;

  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
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
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
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

 public:
  // Constructor, store seen variables
  explicit PrintOpLowering(MLIRContext *context,
                           std::map<std::string, mlir::Value> &v)
      : ConversionPattern(mlir::quantum::PrintOp::getOperationName(), 1,
                          context),
        variables(v) {}

  // Match any Operation that is the QallocOp
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
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
      }

      args.push_back(o);
    }
    rewriter.create<mlir::CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                                  llvm::makeArrayRef(args));
    rewriter.eraseOp(op);

    // parentModule.dump();
    return success();
  }
};

class StdAtanOpLowering : public ConversionPattern {
 private:
  static FlatSymbolRefAttr getOrInsertAtanFunction(PatternRewriter &rewriter,
                                                   ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("atan"))
      return mlir::SymbolRefAttr::get("atan", context);

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto ret_type = rewriter.getF64Type();
    auto arg_type = rewriter.getF64Type();
    auto llvmFnType = LLVM::LLVMFunctionType::get(ret_type, arg_type, false);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "atan", llvmFnType);
    return mlir::SymbolRefAttr::get("atan", context);
  }

 public:
  // Constructor, store seen variables
  explicit StdAtanOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::AtanOp::getOperationName(), 1, context) {}

  // Match any Operation that is the QallocOp
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Local Declarations, get location, parentModule
    // and the context
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();
    auto atan = cast<mlir::AtanOp>(op);

    auto atanRef = getOrInsertAtanFunction(rewriter, parentModule);

    auto call = rewriter.create<mlir::LLVM::CallOp>(loc, rewriter.getF64Type(),
                                                    atanRef, atan.operand());

    rewriter.replaceOp(op, call.getResult(0));

    return success();
  }
};
}  // namespace
namespace qcor {
void QuantumToLLVMLoweringPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}

struct QuantumLLVMTypeConverter : public LLVMTypeConverter {
 private:
  Type convertOpaqueQuantumTypes(OpaqueType type) {
    if (type.getTypeData() == "Qubit") {
      return LLVM::LLVMPointerType::get(get_quantum_type("Qubit", context));
    } else if (type.getTypeData() == "ArgvType") {
      return LLVM::LLVMPointerType::get(
          LLVM::LLVMPointerType::get(IntegerType::get(context, 8)));
    } else if (type.getTypeData() == "qreg") {
      return LLVM::LLVMPointerType::get(get_quantum_type("qreg", context));
    } else if (type.getTypeData() == "Array") {
      return LLVM::LLVMPointerType::get(get_quantum_type("Array", context));
    }
    std::cout << "ERROR WE DONT KNOW WAHT THIS TYPE IS\n";
    return mlir::IntegerType::get(context, 64);
  }

  mlir::MLIRContext *context;

 public:
  QuantumLLVMTypeConverter(mlir::MLIRContext *ctx)
      : LLVMTypeConverter(ctx), context(ctx) {
    addConversion(
        [&](OpaqueType type) { return convertOpaqueQuantumTypes(type); });
  }
};

void QuantumToLLVMLoweringPass::runOnOperation() {
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  QuantumLLVMTypeConverter typeConverter(&getContext());

  OwningRewritePatternList patterns;
  patterns.insert<StdAtanOpLowering>(&getContext());

  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  // Common variables to share across converteres
  std::map<std::string, mlir::Value> variables;
  std::map<mlir::Operation *, std::string> qubit_extract_map;

  patterns.insert<CreateStringLiteralOpLowering>(&getContext(), variables);
  patterns.insert<PrintOpLowering>(&getContext(), variables);

  patterns.insert<QallocOpLowering>(&getContext(), variables);
  patterns.insert<InstOpLowering>(&getContext(), variables, qubit_extract_map,
                                  function_names);
  patterns.insert<SetQregOpLowering>(&getContext(), variables);
  patterns.insert<ExtractQubitOpConversion>(&getContext(), typeConverter,
                                            variables, qubit_extract_map);
  patterns.insert<DeallocOpLowering>(&getContext(), variables);
  patterns.insert<QRTInitOpLowering>(&getContext(), variables);
  patterns.insert<QRTFinalizeOpLowering>(&getContext(), variables);
  patterns.insert<QubitArrayAllocOpLowering>(&getContext(), variables);
  patterns.insert<AssignQubitOpConversion>(&getContext(), variables);

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
}  // namespace qcor