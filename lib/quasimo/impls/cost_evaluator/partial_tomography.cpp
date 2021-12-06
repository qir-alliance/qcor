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
#include "partial_tomography.hpp"

#include "qsim_utils.hpp"
#include "xacc.hpp"

namespace qcor {
namespace QuaSiMo {
double PartialTomoObjFuncEval::evaluate(
    std::shared_ptr<CompositeInstruction> state_prep) {
  auto subKernels = qcor::__internal__::observe(*target_operator, state_prep);
  // Run the pass manager (optimization + placement)
  executePassManager(subKernels);
  // Set qreg size to the bigger of target_operator bits or state prep physical bits
  auto tmp_buffer =
      qalloc(target_operator->nBits() > state_prep->nPhysicalBits()
                 ? target_operator->nBits()
                 : state_prep->nPhysicalBits());

  std::vector<decltype(subKernels[0]->as_xacc())> xaccComposites;
  for (auto &circ : subKernels) {
    xaccComposites.emplace_back(circ->as_xacc());
  }

  xacc::internal_compiler::execute(tmp_buffer.results(), xaccComposites);
  const double energy =
      tmp_buffer.weighted_sum(std::dynamic_pointer_cast<xacc::Observable>(
                                  target_operator->get_as_opaque())
                                  .get());
  return energy;
}

std::vector<double> PartialTomoObjFuncEval::evaluate(
    std::vector<std::shared_ptr<CompositeInstruction>> state_prep_circuits) {
  xacc::info("Batching " + std::to_string(state_prep_circuits.size()) +
             " kernel observable evaluations.");
  std::vector<size_t> nbSubKernelsPerObs;
  std::vector<std::shared_ptr<CompositeInstruction>> fsToExec;
  for (auto &circ : state_prep_circuits) {
    auto subKernels =
        qcor::__internal__::observe(*target_operator, circ);
    // Run the pass manager (optimization + placement)
    executePassManager(subKernels);
    // Track number of obs. kernels
    nbSubKernelsPerObs.emplace_back(subKernels.size());
    fsToExec.insert(fsToExec.end(), subKernels.begin(), subKernels.end());
  }
  auto tmp_buffer = qalloc(target_operator->nBits());

  std::vector<decltype(fsToExec[0]->as_xacc())> fsToExecCasted;
  for (auto &f : fsToExec) {
    fsToExecCasted.emplace_back(f->as_xacc());
  }

  // Execute all kernels:
  xacc::internal_compiler::execute(tmp_buffer.results(), fsToExecCasted);
  size_t bufferCounter = 0;
  std::vector<double> result;
  auto allChildBuffers = tmp_buffer.results()->getChildren();
  // Segregates the child buffers into groups (of each obs eval.)
  for (const auto &nbChildBuffers : nbSubKernelsPerObs) {
    auto temp_buf = qalloc(tmp_buffer.size());
    for (size_t idx = 0; idx < nbChildBuffers; ++idx) {
      auto bufferToAppend = allChildBuffers[bufferCounter + idx];
      temp_buf.results()->appendChild(bufferToAppend->name(), bufferToAppend);
    }
    bufferCounter += nbChildBuffers;
    result.emplace_back(
        temp_buf.weighted_sum(std::dynamic_pointer_cast<xacc::Observable>(
                                  target_operator->get_as_opaque())
                                  .get()));
  }

  assert(result.size() == state_prep_circuits.size());
  return result;
}
}  // namespace QuaSiMo
}  // namespace qcor