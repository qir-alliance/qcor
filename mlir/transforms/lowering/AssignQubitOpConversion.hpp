#pragma once
#include "quantum_to_llvm.hpp"

namespace qcor {
class AssignQubitOpConversion : public ConversionPattern {
protected:
  std::map<std::string, mlir::Value> &variables;

public:
  // CTor: store seen variables
  explicit AssignQubitOpConversion(MLIRContext *context,
                                   std::map<std::string, mlir::Value> &vars)
      : ConversionPattern(mlir::quantum::AssignQubitOp::getOperationName(), 1,
                          context),
        variables(vars) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace qcor