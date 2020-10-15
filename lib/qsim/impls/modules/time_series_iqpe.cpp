#include "qsim_impl.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"

/// Implements time-series phase estimation protocol:
/// Refs: 
/// https://arxiv.org/pdf/2010.02538.pdf

/// Notes:
/// We support both *unverified* and *verified* versions of the protocol.
/// i.e. for noise-free simulation, we can use the *unverified* version whereby
/// no post-selection based on measurement results of the main qubit register is required.

namespace qcor {
namespace qsim {
double PhaseEstimationObjFuncEval::evaluate(
    std::shared_ptr<CompositeInstruction> state_prep) {
  // TODO:
  return 0.0;
}
} // namespace qsim
} // namespace qcor
