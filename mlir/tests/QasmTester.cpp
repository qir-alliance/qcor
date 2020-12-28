#pragma GCC diagnostic ignored "-Wsuggest-override"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wpessimizing-move"

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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
#include "optimization/simplify.hpp"
#include "quantum_dialect.hpp"
#include "staq_parser.hpp"
#include "transformations/desugar.hpp"
#include "transformations/inline.hpp"
#include "transformations/oracle_synthesizer.hpp"

using namespace mlir;
using namespace staq;
std::map<std::string, std::string> inst_map {{"cx", "cnot"}, {"measure", "mz"}};

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

    // Notify the rewriter that this operation has been removed.
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

 public:
  explicit InstOpLowering(MLIRContext *context,
                          std::map<std::string, mlir::Value> &vars)
      : ConversionPattern(mlir::quantum::InstOp::getOperationName(), 1,
                          context),
        variables(vars) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();

    // First goal, get symbol for __quantum__rt__array_get_element_ptr_1d function
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

    // Now get Instruction name and the bits it operates on with qreg names
    auto instOp = cast<mlir::quantum::InstOp>(op);
    auto inst_name = instOp.name().str();
    auto qbits = instOp.qubits();
    auto dense_el_qreg = instOp.qreg_names().getRawStringData();

    std::vector<std::string> qreg_names;
    for (auto el : dense_el_qreg) {
      qreg_names.push_back(el.str());
    }

    // Weird - instOp.qreg_names() acts like a set, can't have 
    // ["q", "q"] for cnot q[0], q[1] for example - its just ["q"]
    // fix that here
    if (qreg_names.size() < qbits.size()) {
      for (int i = qreg_names.size(); i < qbits.size(); i++) {
        qreg_names.push_back(dense_el_qreg[0].str());
      }
    }

    // Get the qbit elements from array as Values
    std::vector<mlir::Value> qbit_values;
    for (int i = 0; i < qbits.size(); i++) {
      auto qbit = qbits.getValue<int64_t>(i);
      auto qreg_name = qreg_names[i];

      // Create LLVM ConstantOp for qubit index
      Value qbit_idx = rewriter.create<LLVM::ConstantOp>(
          loc, LLVM::LLVMType::getInt64Ty(rewriter.getContext()),
          rewriter.getIntegerAttr(rewriter.getI64Type(), qbit));

      // Make sure the qreg name is in the seen allocated qresg
      if (!variables.count(qreg_name)) {
        std::cout << "Error, " << qreg_name << " not allocated.\n";
        // return -1;
      }

      // Get the pre-allocated qreg
      auto qbit_array = variables[qreg_name];

      // Construct the __quantum__rt__array_get_element_ptr_1d CallOp
      // should be Qubit* __quantum__rt__array_get_element_ptr_1d(Array*, QubitIdx)
      auto array_qbit_type = LLVM::LLVMType::getInt64Ty(context).getPointerTo();
      auto get_qbit_qir_call = rewriter.create<mlir::CallOp>(
          loc, symbol_ref, array_qbit_type,
          ArrayRef<Value>({qbit_array, qbit_idx}));

      // Store the qubit value that was returned by this function
      qbit_values.push_back(get_qbit_qir_call.getResult(0));
    }

    // Need to find the quantum instruction function
    // Should be void __quantum__qis__INST(Qubit q) for example
    FlatSymbolRefAttr q_symbol_ref;
    std::string q_function_name = "__quantum__qis__" + (inst_map.count(inst_name) ? inst_map[inst_name] : inst_name);
    if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(q_function_name)) {
      q_symbol_ref = SymbolRefAttr::get(q_function_name, context);
    } else {
      auto void_type = LLVM::LLVMType::getVoidTy(context);

      // Need a Int64Type for each qubit argument
      std::vector<LLVM::LLVMType> tmp_arg_types;
      for (int i = 0; i < qbits.size(); i++) {
        auto qubit_index_type = LLVM::LLVMType::getInt64Ty(context).getPointerTo();
        tmp_arg_types.push_back(qubit_index_type);
      }

      // FIXME loop over params too to add double types

      // Create void (int, int) or void (int)
      auto get_ptr_qbit_ftype = LLVM::LLVMType::getFunctionTy(
          void_type, llvm::makeArrayRef(tmp_arg_types), true);

      //Insert the function since it hasn't been seen yet
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(parentModule->getLoc(), q_function_name,
                                        get_ptr_qbit_ftype);

      q_symbol_ref =
          mlir::SymbolRefAttr::get(q_function_name, context);
    }

    // Now create the CallOp for __quantum__qis__INST(Qubit q)
    // std::vector<mlir::Value> bitcast_and_loaded;
    // auto bitcast_type = LLVM::LLVMType::getInt64Ty(context).getPointerTo().getPointerTo();
    // for (auto qbit_value : qbit_values) {
    //   auto result = rewriter.create<LLVM::BitcastOp>(loc, bitcast_type, qbit_value);
    //   auto tmp = rewriter.create<LLVM::LoadOp>(loc, LLVM::LLVMType::getInt64Ty(context).getPointerTo(), result);
    //   bitcast_and_loaded.push_back(tmp;)
    // }

    auto void_type = LLVM::LLVMType::getVoidTy(context);
    auto qinst_qir_call = rewriter.create<mlir::CallOp>(
        loc, q_symbol_ref, void_type, llvm::makeArrayRef(qbit_values));
    
    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);

    return success();
  }
};

struct ReturnOpLowering : public OpRewritePattern<mlir::quantum::ReturnOp> {
  using OpRewritePattern<mlir::quantum::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::quantum::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlined.
    if (op.hasOperand()) return failure();

    // We lower "toy.return" directly to "std.return".
    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op);
    return success();
  }
};

struct QuantumToLLVMLoweringPass
    : public PassWrapper<QuantumToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() final;

 public:
  QuantumToLLVMLoweringPass(std::map<std::string, mlir::Value> &vars)
      : variables(vars) {}

 protected:
  std::map<std::string, mlir::Value> &variables;
};

void QuantumToLLVMLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering. For this lowering, we are only targeting
  // the LLVM dialect.
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  // During this lowering, we will also be lowering the MemRef types, that are
  // currently being operated on, to a representation in LLVM. To perform this
  // conversion we use a TypeConverter as part of the lowering. This converter
  // details how one type maps to another. This is necessary now that we will be
  // doing more complicated lowerings, involving loop region arguments.
  LLVMTypeConverter typeConverter(&getContext());

  // Now that the conversion target has been defined, we need to provide the
  // patterns used for lowering. At this point of the compilation process, we
  // have a combination of `toy`, `affine`, and `std` operations. Luckily, there
  // are already exists a set of patterns to transform `affine` and `std`
  // dialects. These patterns lowering in multiple stages, relying on transitive
  // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
  // patterns must be applied to fully transform an illegal operation into a
  // set of legal ones.
  OwningRewritePatternList patterns;
  // populateAffineToStdConversionPatterns(patterns, &getContext());
  // populateLoopToStdConversionPatterns(patterns, &getContext());
  populateStdToLLVMConversionPatterns(typeConverter, patterns);
  patterns.insert<ReturnOpLowering>(&getContext());

  patterns.insert<QallocOpLowering>(&getContext(), variables);
  patterns.insert<InstOpLowering>(&getContext(), variables);

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

int main(int argc, char **argv) {
  std::string lineText = R"#(OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
creg c[2];
measure q -> c;
)#";

  std::cout << "Original:\n" << lineText << "\n";
  ast::ptr<ast::Program> prog;
  try {
    prog = parser::parse_string(lineText);
    transformations::desugar(*prog);
    transformations::synthesize_oracles(*prog);
  } catch (std::exception &e) {
    std::stringstream ss;
    std::cout << e.what() << "\n";
  }

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::quantum::QuantumDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();

  qasm_parser::StaqToMLIR visitor(context);
  visitor.visit(*prog);

  visitor.addReturn();

  std::cout << "MLIR + Quantum Dialect:\n";
  visitor.module()->dump();

  mlir::PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  // applyPassManagerCLOptions(pm);
  // pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());

  // Finish lowering the toy IR to the LLVM dialect.
  std::map<std::string, mlir::Value> allocated_variables;
  pm.addPass(std::make_unique<QuantumToLLVMLoweringPass>(allocated_variables));

  auto module = visitor.module();
  auto module_op = module.getOperation();
  pm.run(module_op);

  std::cout << "Lowered to LLVM MLIR Dialect:\n";
  module_op->dump();
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);

  std::cout << "Lowered to LLVM IR:\n";
  llvmModule->dump();

  return 0;
}
