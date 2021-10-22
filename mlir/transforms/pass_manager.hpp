/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#pragma once
#include "mlir/Dialect/Affine/Passes.h"
#include "optimizations/IdentityPairRemovalPass.hpp"
#include "optimizations/PermuteGatePass.hpp"
#include "optimizations/RemoveUnusedQIRCallsPass.hpp"
#include "optimizations/RotationMergingPass.hpp"
#include "optimizations/SimplifyQubitExtractPass.hpp"
#include "optimizations/SingleQubitGateMergingPass.hpp"
#include "optimizations/CphaseRotationMergingPass.hpp"
#include "optimizations/ModifierBlockInliner.hpp"
#include "quantum_to_llvm.hpp"
#include "lowering/ModifierRegionLowering.hpp"

// Construct QCOR MLIR pass manager:
// Make sure we use the same set of passes and configs
// across different use cases of MLIR compilation.
namespace qcor {
void configureOptimizationPasses(mlir::PassManager &passManager) {
  // Try inline both before and after loop unroll.
  passManager.addPass(mlir::createInlinerPass());
  passManager.addPass(std::make_unique<ModifierBlockInlinerPass>());
  auto loop_unroller = mlir::createLoopUnrollPass(/*unrollFactor*/-1, /*unrollUpToFactor*/ false, /*unrollFull*/true);
  // Nest a pass manager that operates on functions within the one which
  // operates on ModuleOp.
  OpPassManager &nestedFunctionPM = passManager.nest<mlir::FuncOp>();
  nestedFunctionPM.addPass(std::move(loop_unroller));
  passManager.addPass(mlir::createInlinerPass());

  passManager.addPass(std::make_unique<SimplifyQubitExtractPass>());
  // TODO: configure the pass pipeline to handle repeated applications of
  // passes. Add passes
  constexpr int N_REPS = 10;
  for (int i = 0; i < N_REPS; ++i) {
    // Simple Identity pair removals
    passManager.addPass(std::make_unique<SingleQubitIdentityPairRemovalPass>());
    passManager.addPass(std::make_unique<CNOTIdentityPairRemovalPass>());
    passManager.addPass(std::make_unique<DuplicateResetRemovalPass>());
    
    // Rotation merging
    passManager.addPass(std::make_unique<RotationMergingPass>());
    passManager.addPass(std::make_unique<CPhaseRotationMergingPass>());

    // General gate sequence re-synthesize
    passManager.addPass(std::make_unique<SingleQubitGateMergingPass>());
    // Try permute gates to realize more merging opportunities
    passManager.addPass(std::make_unique<PermuteGatePass>());
  }

  // Remove dead code
  passManager.addPass(std::make_unique<RemoveUnusedQIRCallsPass>());
}
} // namespace qcor