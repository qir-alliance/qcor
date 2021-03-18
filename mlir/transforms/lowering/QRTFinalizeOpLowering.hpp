#pragma once
#include "quantum_to_llvm.hpp"

namespace qcor {
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
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace qcor