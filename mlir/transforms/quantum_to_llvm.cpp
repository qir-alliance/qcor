#include "quantum_to_llvm.hpp"
#include "lowering/AssignQubitOpConversion.hpp"
#include "lowering/CreateStringLiteralOpLowering.hpp"
#include "lowering/DeallocOpLowering.hpp"
#include "lowering/ExtractQubitOpConversion.hpp"
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
#include <iostream>

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
  patterns.insert<QarraySliceOpLowering>(&getContext(), variables);
  patterns.insert<QarrayConcatOpLowering>(&getContext(), variables);

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
} // namespace qcor