#pragma once
#include "quantum_to_llvm.hpp"

namespace qcor {
// The goal of InstOpLowering is to convert all QuantumDialect
// InstOp (quantum.inst) to the corresponding __quantum__qis__INST(int64*, ...)
// call
class ValueSemanticsInstOpLowering : public ConversionPattern {
protected:

  std::vector<std::string> &module_function_names;
  mutable std::map<std::string, std::string> inst_map{{"cx", "cnot"},
                                                      {"measure", "mz"}};

public:
  // The Constructor, store the variables and qubit extract op map
  explicit ValueSemanticsInstOpLowering(MLIRContext *context,
                          std::vector<std::string> &f_names)
      : ConversionPattern(mlir::quantum::ValueSemanticsInstOp::getOperationName(), 1,
                          context),
       
        module_function_names(f_names) {}

  // Match and replace all InstOps
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace qcor