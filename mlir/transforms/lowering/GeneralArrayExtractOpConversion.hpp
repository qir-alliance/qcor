#pragma once
#include "quantum_to_llvm.hpp"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"

namespace qcor {
// The goal of this OpConversion is to map vector.extract on a
// qalloc qubit vector to the MSFT QIR __quantum__rt__array_get_element_ptr_1d()
// call
class GeneralArrayExtractOpConversion : public ConversionPattern {
protected:
  LLVMTypeConverter &typeConverter;
  inline static const std::string qir_get_qubit_from_array =
      "__quantum__rt__array_get_element_ptr_1d";
  std::map<std::string, mlir::Value> &vars;
  std::map<mlir::Operation *, std::string> &qubit_extract_map;

public:
  explicit GeneralArrayExtractOpConversion(
      MLIRContext *context, LLVMTypeConverter &c,
      std::map<std::string, mlir::Value> &v,
      std::map<mlir::Operation *, std::string> &qem)
      : ConversionPattern(mlir::quantum::GeneralArrayExtractOp::getOperationName(), 1,
                          context),
        typeConverter(c), vars(v), qubit_extract_map(qem) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace qcor