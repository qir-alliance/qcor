/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "quantum_kernel.hpp"

#include "CompositeInstruction.hpp"
#include "IRProvider.hpp"
#include "Instruction.hpp"

namespace qcor {
namespace internal {
bool is_not_measure(std::shared_ptr<xacc::Instruction> inst) {
  return inst->name() != "Measure";
}

std::vector<std::shared_ptr<xacc::Instruction>> handle_adjoint_instructions(
    std::vector<std::shared_ptr<xacc::Instruction>> instructions,
    std::shared_ptr<CompositeInstruction> tempKernel) {
  auto provider = qcor::__internal__::get_provider();
  for (int i = 0; i < instructions.size(); i++) {
    auto inst = tempKernel->getInstruction(i);
    // Parametric gates:
    if (inst->name() == "Rx" || inst->name() == "Ry" || inst->name() == "Rz" ||
        inst->name() == "CPhase" || inst->name() == "U1" ||
        inst->name() == "CRZ") {
      inst->setParameter(0, -inst->getParameter(0).template as<double>());
    }
    // Handles T and S gates, etc... => T -> Tdg
    else if (inst->name() == "T") {
      // Forward along the buffer name
      auto tdg = provider->createInstruction(
          "Tdg", {std::make_pair(inst->getBufferName(0), inst->bits()[0])});
      tempKernel->replaceInstruction(i, tdg);
    } else if (inst->name() == "S") {
      auto sdg = provider->createInstruction(
          "Sdg", {std::make_pair(inst->getBufferName(0), inst->bits()[0])});
      tempKernel->replaceInstruction(i, sdg);
    } else if (inst->name() == "Tdg") {
      // Forward along the buffer name
      auto t = provider->createInstruction(
          "T", {std::make_pair(inst->getBufferName(0), inst->bits()[0])});
      tempKernel->replaceInstruction(i, t);
    } else if (inst->name() == "Sdg") {
      auto s = provider->createInstruction(
          "S", {std::make_pair(inst->getBufferName(0), inst->bits()[0])});
      tempKernel->replaceInstruction(i, s);
    }
  }

  // We update/replace instructions in the derived.parent_kernel composite,
  // hence collecting these new instructions and reversing the sequence.
  auto new_instructions = tempKernel->getInstructions();
  std::reverse(new_instructions.begin(), new_instructions.end());

  // Are we in a compute section?
  // Make sure we mark all instructions appropriately.
  // e.g. we do create new instructions, e.g. Tdg, Sdg, in the above step.
  if (::quantum::qrt_impl->isComputeSection()) {
    for (auto &inst : new_instructions) {
      inst->attachMetadata({{"__qcor__compute__segment__", true}});
    }
  }
  return new_instructions;
}
}  // namespace internal
}  // namespace qcor