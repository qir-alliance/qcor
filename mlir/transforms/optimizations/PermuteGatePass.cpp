#include "PermuteGatePass.hpp"
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

namespace qcor {
void PermuteGatePass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}

void PermuteGatePass::runOnOperation() {
  // Walk the operations within the function.
  std::vector<mlir::quantum::ValueSemanticsInstOp> deadOps;
  getOperation().walk([&](mlir::quantum::ValueSemanticsInstOp op) {
    auto inst_name = op.name();
    // Move "Rz" forward
    if (inst_name.str() == "rz") {
      auto return_value = *op.result().begin();
      if (return_value.hasOneUse()) {
        // get that one user
        auto user = *return_value.user_begin();
        // cast to a inst op
        if (auto next_inst =
                dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(user)) {
          if (next_inst.name() == "cx" || next_inst.name() == "cnot") {
            if (next_inst.getOperand(0) == op.result().front()) {
              mlir::OpBuilder rewriter(op);
              // rz connect to control bit (operand 0)
              // Permute rz:
              rewriter.setInsertionPointAfter(next_inst);
              mlir::Value cx_ctrl_out = next_inst.result().front();
              auto new_rz_inst =
                  rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                      op.getLoc(),
                      llvm::makeArrayRef({op.getOperand(0).getType()}), "rz",
                      llvm::makeArrayRef(cx_ctrl_out),
                      llvm::makeArrayRef(op.getOperand(1)));

              // Input to original rz => cnot
              next_inst.getOperand(0).replaceAllUsesWith(op.getOperand(0));
              // First output of cx (control line) to output of the new rz
              // except the new rz which is connected to the output of cx
              cx_ctrl_out.replaceAllUsesExcept(
                  new_rz_inst.result().front(),
                  mlir::SmallPtrSet<Operation *, 1>{new_rz_inst});
              deadOps.emplace_back(op);
            }
          }
        }
      }
    }
  });

  for (auto &op : deadOps) {
    op->dropAllUses();
    op.erase();
  }
}
} // namespace qcor