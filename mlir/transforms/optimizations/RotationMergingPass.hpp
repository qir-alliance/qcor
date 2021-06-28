#pragma once
#include "Quantum/QuantumOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace qcor {
// We make each optimization routine into its own Pass so that
// option such as `--print-ir-before-all` can be used to inspect
// each pass independently.
// TODO: make this a FunctionPass
struct RotationMergingPass
    : public PassWrapper<RotationMergingPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() final;
  RotationMergingPass() {}

private:
  // Use a simple list of pairs to handle bi-directional check.
  // This list is quite short so performance shouldn't be an issue!
  static inline const std::vector<std::pair<std::string, std::string>> search_gates{
      {"rx", "rx"}, {"ry", "ry"}, {"rz", "rz"},
      {"x", "rx"},  {"y", "ry"},  {"z", "rz"}};
  // Angle that we considered zero:
  static constexpr double ZERO_ROTATION_TOLERANCE = 1e-9;
  bool should_combine(const std::string &name1, const std::string &name2) const;
};
} // namespace qcor