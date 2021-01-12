#include "partial_tomography.hpp"
#include "qsim_utils.hpp"
#include "xacc.hpp"

namespace qcor {
namespace QuaSiMo {
double PartialTomoObjFuncEval::evaluate(
    std::shared_ptr<CompositeInstruction> state_prep) {
  auto subKernels = qcor::__internal__::observe(
      xacc::as_shared_ptr(target_operator), state_prep);
  // Run the pass manager (optimization + placement)
  executePassManager(subKernels);
  auto tmp_buffer = qalloc(state_prep->nPhysicalBits());
  xacc::internal_compiler::execute(tmp_buffer.results(), subKernels);
  const double energy = tmp_buffer.weighted_sum(target_operator);
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
        qcor::__internal__::observe(xacc::as_shared_ptr(target_operator), circ);
    // Run the pass manager (optimization + placement)
    executePassManager(subKernels);
    // Track number of obs. kernels
    nbSubKernelsPerObs.emplace_back(subKernels.size());
    fsToExec.insert(fsToExec.end(), subKernels.begin(), subKernels.end());
  }
  auto tmp_buffer = qalloc(target_operator->nBits());
  // Execute all kernels:
  xacc::internal_compiler::execute(tmp_buffer.results(), fsToExec);
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
    result.emplace_back(temp_buf.weighted_sum(target_operator));
  }

  assert(result.size() == state_prep_circuits.size());
  return result;
}
} // namespace QuaSiMo
} // namespace qcor