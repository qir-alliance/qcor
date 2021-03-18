#pragma once
#include "quantum_to_llvm.hpp"

namespace qcor {
class CreateStringLiteralOpLowering : public ConversionPattern {
private:
  std::map<std::string, mlir::Value> &variables;

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module);

public:
  // Constructor, store seen variables
  explicit CreateStringLiteralOpLowering(MLIRContext *context,
                                         std::map<std::string, mlir::Value> &v)
      : ConversionPattern(
            mlir::quantum::CreateStringLiteralOp::getOperationName(), 1,
            context),
        variables(v) {}

  // Match any Operation that is the QallocOp
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace qcor