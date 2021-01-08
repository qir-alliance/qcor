#include "vqe.hpp"
#include "qsim_utils.hpp"

namespace qcor {
namespace QuaSiMo {
bool VqeWorkflow::initialize(const HeterogeneousMap &params) {
  const std::string DEFAULT_OPTIMIZER = "nlopt";
  optimizer.reset();
  if (params.pointerLikeExists<Optimizer>("optimizer")) {
    optimizer =
        xacc::as_shared_ptr(params.getPointerLike<Optimizer>("optimizer"));
  } else {
    optimizer = createOptimizer(DEFAULT_OPTIMIZER);
  }
  config_params = params;
  // VQE workflow requires an optimizer
  return (optimizer != nullptr);
}

QuantumSimulationResult
VqeWorkflow::execute(const QuantumSimulationModel &model) {
  // If the model includes a concrete variational ansatz:
  if (model.user_defined_ansatz) {
    auto nParams = model.user_defined_ansatz->nParams();
    evaluator = getEvaluator(model.observable, config_params);

    OptFunction f(
        [&](const std::vector<double> &x, std::vector<double> &dx) {
          auto kernel = model.user_defined_ansatz->evaluate_kernel(x);
          auto energy = evaluator->evaluate(kernel);
          return energy;
        },
        nParams);

    auto result = optimizer->optimize(f);
    // std::cout << "Min energy = " << result.first << "\n";
    return {{"energy", result.first}, {"opt-params", result.second}};
  }

  // TODO: support ansatz generation methods:
  return {};
}
} // namespace QuaSiMo
} // namespace qcor