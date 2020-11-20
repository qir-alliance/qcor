#include "partial_tomography.hpp"
#include "qsim_utils.hpp"

namespace qcor {
namespace qsim {
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
} // namespace qsim
} // namespace qcor