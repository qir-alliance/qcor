#include "quantum_to_llvm.hpp"

#include <iostream>

#include "lowering/AdjointURegionLowering.hpp"
#include "lowering/AssignQubitOpConversion.hpp"
#include "lowering/CreateStringLiteralOpLowering.hpp"
#include "lowering/CtrlURegionLowering.hpp"
#include "lowering/DeallocOpLowering.hpp"
#include "lowering/ExtractQubitOpConversion.hpp"
#include "lowering/GeneralArrayExtractOpConversion.hpp"
#include "lowering/InstOpLowering.hpp"
#include "lowering/PowURegionLowering.hpp"
#include "lowering/PrintOpLowering.hpp"
#include "lowering/QRTFinalizeOpLowering.hpp"
#include "lowering/QRTInitOpLowering.hpp"
#include "lowering/QallocOpLowering.hpp"
#include "lowering/QarrayConcatOpLowering.hpp"
#include "lowering/QarraySliceOpLowering.hpp"
#include "lowering/QubitArrayAllocOpLowering.hpp"
#include "lowering/SetQregOpLowering.hpp"
#include "lowering/StdAtanOpLowering.hpp"
#include "lowering/ValueSemanticsInstOpLowering.hpp"
#include "optimizations/IdentityPairRemovalPass.hpp"
#include "optimizations/RemoveUnusedQIRCalls.hpp"

namespace qcor {
mlir::Type get_quantum_type(std::string type, mlir::MLIRContext *context) {
  return LLVM::LLVMStructType::getOpaque(type, context);
}

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
    exit(0);
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
  auto module = getOperation();

  if (q_optimizations) {
    // TODO Figure out how to rip this out to its on MLIR-level Pass. 
    // I'm struggling to make that happen...
    
    // First, add any Optimization Passes.
    // We note that some opt passes will free up other optimizations that
    // would otherwise be missed on the first pass, so do this a certain
    // number of times.
    int n_heuristic_passes = 5;
    for (int i = 0; i < n_heuristic_passes; i++) {
      patterns.insert<SingleQubitIdentityPairRemovalPattern>(&getContext());
      patterns.insert<CNOTIdentityPairRemovalPattern>(&getContext());
      if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
    }

    // Clean up...
    patterns.insert<RemoveUnusedExtractQubitCalls>(&getContext());
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
    patterns.insert<RemoveUnusedQallocCalls>(&getContext());
  }

  // Lower arctan correctly
  patterns.insert<StdAtanOpLowering>(&getContext());

  // Add Standard to LLVM
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  // Common variables to share across converteres
  std::map<std::string, mlir::Value> variables;
  std::map<mlir::Operation *, std::string> qubit_extract_map;

  patterns.insert<CreateStringLiteralOpLowering>(&getContext(), variables);
  patterns.insert<PrintOpLowering>(&getContext(), variables);

  patterns.insert<QallocOpLowering>(&getContext(), variables);
  patterns.insert<InstOpLowering>(&getContext(), variables, qubit_extract_map,
                                  function_names);
  patterns.insert<ValueSemanticsInstOpLowering>(&getContext(), function_names);
  patterns.insert<SetQregOpLowering>(&getContext(), variables);
  patterns.insert<ExtractQubitOpConversion>(&getContext(), typeConverter,
                                            variables, qubit_extract_map);
  patterns.insert<GeneralArrayExtractOpConversion>(
      &getContext(), typeConverter, variables, qubit_extract_map);
  patterns.insert<DeallocOpLowering>(&getContext(), variables);
  patterns.insert<QRTInitOpLowering>(&getContext(), variables);
  patterns.insert<QRTFinalizeOpLowering>(&getContext(), variables);
  patterns.insert<QubitArrayAllocOpLowering>(&getContext(), variables);
  patterns.insert<AssignQubitOpConversion>(&getContext(), variables);
  patterns.insert<QarraySliceOpLowering>(&getContext(), variables);
  patterns.insert<QarrayConcatOpLowering>(&getContext(), variables);
  patterns.insert<StartPowURegionOpLowering>(&getContext());
  patterns.insert<EndPowURegionOpLowering>(&getContext());
  patterns.insert<StartAdjointURegionOpLowering>(&getContext());
  patterns.insert<EndAdjointURegionOpLowering>(&getContext());
  patterns.insert<StartCtrlURegionOpLowering>(&getContext());
  patterns.insert<EndCtrlURegionOpLowering>(&getContext());

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

// std::unique_ptr<mlir::Pass> createQuantumOptPass() {return
// std::make_unique<IdentityPairRemovalPass>();}

}  // namespace qcor