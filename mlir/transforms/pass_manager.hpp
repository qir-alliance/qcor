#pragma once 
#include "optimizations/RotationMergingPass.hpp"
#include "optimizations/SingleQubitGateMergingPass.hpp"
#include "optimizations/IdentityPairRemovalPass.hpp"
#include "optimizations/RemoveUnusedQIRCallsPass.hpp"
#include "quantum_to_llvm.hpp"

// Construct QCOR MLIR pass manager:
// Make sure we use the same set of passes and configs
// across different use cases of MLIR compilation.
namespace qcor {
void configureOptimizationPasses(mlir::PassManager &passManager) {
  // TODO: configure the pass pipeline to handle repeated applications of passes.
  // Add passes
  // Simple Identity pair removals
  passManager.addPass(std::make_unique<SingleQubitIdentityPairRemovalPass>());
  passManager.addPass(std::make_unique<CNOTIdentityPairRemovalPass>());

  // Rotation merging
  passManager.addPass(std::make_unique<RotationMergingPass>());
  // General gate sequence re-synthesize
  passManager.addPass(std::make_unique<SingleQubitGateMergingPass>());


  // Remove dead code
  passManager.addPass(std::make_unique<RemoveUnusedQIRCallsPass>());
}
} // namespace qcor