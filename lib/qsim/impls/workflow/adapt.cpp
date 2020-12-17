#include "qsim_utils.hpp"
#include "adapt.hpp"

namespace qcor {
namespace qsim {
bool AdaptVqeWorkflow::initialize(const HeterogeneousMap &params) {
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
AdaptVqeWorkflow::execute(const QuantumSimulationModel &model) {
  return {};
}
} // namespace qsim
} // namespace qcor