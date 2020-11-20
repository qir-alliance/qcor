#include "qaoa.hpp"

namespace qcor {
namespace qsim {
bool QaoaWorkflow::initialize(const HeterogeneousMap &params) {
  const std::string DEFAULT_OPTIMIZER = "nlopt";
  optimizer.reset();
  if (params.pointerLikeExists<Optimizer>("optimizer")) {
    optimizer =
        xacc::as_shared_ptr(params.getPointerLike<Optimizer>("optimizer"));
  } else {
    optimizer = createOptimizer(DEFAULT_OPTIMIZER);
  }
  config_params = params;
  return (optimizer != nullptr);
}

QuantumSimulationResult
QaoaWorkflow::execute(const QuantumSimulationModel &model) {
  // TODO
  return {};
}
} // namespace qsim
} // namespace qcor