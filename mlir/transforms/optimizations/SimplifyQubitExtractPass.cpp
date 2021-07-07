#include "SimplifyQubitExtractPass.hpp"
#include "Quantum/QuantumOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include <iostream>

namespace qcor {
void SimplifyQubitExtractPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}
void SimplifyQubitExtractPass::runOnOperation() {
  std::vector<mlir::quantum::ExtractQubitOp> unique_extract_ops;
  std::unordered_map<mlir::Operation *,
                     std::unordered_map<int64_t, mlir::Value>>
      extract_qubit_map;

  getOperation().walk([&](mlir::quantum::ExtractQubitOp op) {
    mlir::Value idx_val = op.idx();
    mlir::Value qreg = op.qreg();
    mlir::Operation *qreg_create_op = qreg.getDefiningOp();
    if (qreg_create_op) {
      if (extract_qubit_map.find(qreg_create_op) == extract_qubit_map.end()) {
        extract_qubit_map[qreg_create_op] = {};
      }

      auto idx_def_op = idx_val.getDefiningOp();
      if (idx_def_op) {
        // Try cast:
        if (auto const_def_op =
                dyn_cast_or_null<mlir::ConstantIntOp>(idx_def_op)) {
          // std::cout << "Constant extract index " << const_def_op.getValue()
          //           << "\n";
          const int64_t index_const = const_def_op.getValue();
          auto &previous_qreg_extract = extract_qubit_map[qreg_create_op];
          if (previous_qreg_extract.find(index_const) ==
              previous_qreg_extract.end()) {
            // std::cout << "First use\n";
            previous_qreg_extract[index_const] = op.qbit();
          } else {
            mlir::Value previous_extract = previous_qreg_extract[index_const];
            previous_extract.dump();
            const std::function<mlir::Value(mlir::Value)> get_last_use =
                [&get_last_use](mlir::Value var) -> mlir::Value {
              if (var.hasOneUse()) {
                auto use = *var.user_begin();
                auto next_inst =
                    dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(use);
                if (next_inst) {
                  if (next_inst.qubits().size() == 1) {
                    return get_last_use(*next_inst.result_begin());
                  } else {
                    assert(next_inst.qubits().size() == 2);
                    // std::cout << "Two qubit gate use\n";
                    // Need to determine which operand this value is used
                    // i.e. map to the corresponding output
                    for (size_t i = 0; i < next_inst.qubits().size(); ++i) {
                      mlir::Value operand = next_inst.qubits()[i];
                      if (operand == var) {
                        // std::cout << "Find operand: " << i << "\n";
                        return get_last_use(next_inst.result()[i]);
                      }
                    }
                    // Something wrong, cannot match the operand of 2-q
                    // ValueSemanticsInstOp
                    __builtin_unreachable();
                    assert(false);
                    return var;
                  }
                } else {
                  return var;
                }
              } else {
                // No other use (last)
                // std::cout << "Last use\n";
                // var.dump();
                return var;
              }
            };

            mlir::Value last_use = get_last_use(previous_extract);
            op.qbit().replaceAllUsesWith(last_use);
          }
        }
      }
    }
  });
}
} // namespace qcor