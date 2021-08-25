#pragma once
#include "Quantum/QuantumOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

class SingleQubitIdentityPairRemovalPass
    : public PassWrapper<SingleQubitIdentityPairRemovalPass,
                         OperationPass<ModuleOp>> {
 protected:
  static inline const std::map<std::string, std::string> search_gates{
      {"x", "x"},   {"y", "y"},   {"z", "z"},   {"h", "h"},
      {"t", "tdg"}, {"tdg", "t"}, {"s", "sdg"}, {"sdg", "s"}};
  bool should_remove(std::string name1, std::string name2) const {
    if (search_gates.count(name1)) {
      return search_gates.at(name1) == name2;
    }
    return false;
  }

 public:
  void runOnOperation() final {
    // Walk the operations within the function.
    std::vector<mlir::quantum::ValueSemanticsInstOp> deadOps;

    getOperation().walk([&](mlir::quantum::ValueSemanticsInstOp op) {
      if (std::find(deadOps.begin(), deadOps.end(), op) != deadOps.end()) {
        // Skip this op since it was merged (forward search)
        return;
      }

      auto inst_name = op.name();
      auto return_value = *op.result().begin();
      if (return_value.hasOneUse()) {
        // get that one user
        auto user = *return_value.user_begin();
        // cast to a inst op
        if (auto next_inst =
                dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(user)) {
          // check that it is one of our known id pairs
          if (should_remove(next_inst.name().str(), inst_name.str())) {
            // need to get users of next_inst and point them to use
            // op.getOperands
            (*next_inst.result_begin()).replaceAllUsesWith(op.getOperand(0));
            deadOps.emplace_back(op);
            deadOps.emplace_back(next_inst);
          }
        }
      }
    });

    for (auto &op : deadOps) {
      op->dropAllUses();
      op.erase();
    }
  }
};