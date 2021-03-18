#pragma once
#include "quantum_to_llvm.hpp"

namespace qcor {
class QarraySliceOpLowering : public ConversionPattern {
protected:
  // Constant string for runtime function name
  inline static const std::string qir_qubit_array_slice =
      "__quantum__rt__array_slice";
  // Rudimentary symbol table, seen variables
  std::map<std::string, mlir::Value> &variables;

public:
  // Constructor, store seen variables
  explicit QarraySliceOpLowering(MLIRContext *context,
                                 std::map<std::string, mlir::Value> &vars)
      : ConversionPattern(mlir::quantum::ArraySliceOp::getOperationName(), 1,
                          context),
        variables(vars) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace qcor