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
#include "quantum_to_llvm.hpp"

#include <iostream>

#include "lowering/AssignQubitOpConversion.hpp"
#include "lowering/CreateStringLiteralOpLowering.hpp"
#include "lowering/DeallocOpLowering.hpp"
#include "lowering/ExtractQubitOpConversion.hpp"
#include "lowering/GeneralArrayExtractOpConversion.hpp"
#include "lowering/InstOpLowering.hpp"
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
#include "lowering/CallableLowering.hpp"
#include "lowering/ComputeMarkerLowering.hpp"
#include "lowering/ConditionalOpLowering.hpp"
#include "lowering/ModifierRegionLowering.hpp"

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
    } else if (type.getTypeData() == "Callable") {
      return LLVM::LLVMPointerType::get(get_quantum_type("Callable", context));
    } else if (type.getTypeData() == "Tuple") {
      return LLVM::LLVMPointerType::get(get_quantum_type("Tuple", context));
    } else if (type.getTypeData() == "Result") {
      return LLVM::LLVMPointerType::get(get_quantum_type("Result", context));
    }
    std::cout << "ERROR WE DONT KNOW WHAT THIS TYPE IS: " << std::string(type.getTypeData()) << "\n";
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

  // Lower arctan correctly
  patterns.insert<StdAtanOpLowering>(&getContext());
  // Affine to Standard
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());

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
  patterns.insert<ResultCastOpLowering>(&getContext());
  patterns.insert<IntegerCastOpLowering>(&getContext());
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
  patterns.insert<AdjURegionOpLowering>(&getContext());
  patterns.insert<PowURegionOpLowering>(&getContext());
  patterns.insert<CtrlURegionOpLowering>(&getContext());
  patterns.insert<EndModifierRegionOpLowering>(&getContext());
  patterns.insert<ComputeMarkerOpLowering>(&getContext());
  patterns.insert<ComputeUnMarkerOpLowering>(&getContext());
  patterns.insert<TupleUnpackOpLowering>(&getContext());
  patterns.insert<CreateCallableOpLowering>(&getContext());
  patterns.insert<ConditionalOpLowering>(&getContext());

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

// std::unique_ptr<mlir::Pass> createQuantumOptPass() {return
// std::make_unique<IdentityPairRemovalPass>();}

}  // namespace qcor