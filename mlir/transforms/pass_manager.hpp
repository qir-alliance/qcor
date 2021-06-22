#pragma once 
#include "optimizations/RotationMergingPass.hpp"
#include "optimizations/SingleQubitGateMergingPass.hpp"
#include "quantum_to_llvm.hpp"

// Construct QCOR MLIR pass manager:
// Make sure we use the same set of passes and configs
// across different use cases of MLIR compilation.
namespace qcor {
void configureOptimizationPasses(mlir::PassManager &passManager) {
  // Add passes
  // Rotation merging
  passManager.addPass(std::make_unique<RotationMergingPass>());
  // General gate sequence re-synthesize
  passManager.addPass(std::make_unique<SingleQubitGateMergingPass>());
}
} // namespace qcor