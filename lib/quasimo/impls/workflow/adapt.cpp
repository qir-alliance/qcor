#include "adapt.hpp"
#include "qsim_utils.hpp"
#include "xacc.hpp"

namespace qcor {
namespace QuaSiMo {
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

  if (!params.stringExists("gradient_strategy")) {
    config_params.insert("gradient_strategy", "autodiff");
  }
  
  return (optimizer != nullptr);
}

QuantumSimulationResult
AdaptVqeWorkflow::execute(const QuantumSimulationModel &model) {
  // Note: to make sure compatibility with XACC, we use the same
  // parameters for Adapt as XACC, just forward it to the XACC impl.
  
  // Some extra adapt params comming from the model.
  auto accelerator = xacc::internal_compiler::get_qpu();
  xacc::HeterogeneousMap extra_params{
      {"observable", std::dynamic_pointer_cast<xacc::Observable>(
                         model.observable->get_as_opaque())},
      {"sub-algorithm", "vqe"},
      {"accelerator", accelerator},
      {"optimizer", (*optimizer)->xacc_opt}};

  // If the model contains an ansatz:
  if (model.user_defined_ansatz) {
    std::shared_ptr<xacc::CompositeInstruction> state_prep_circ =
        model.user_defined_ansatz->evaluate_kernel({})->as_xacc();
    extra_params.insert("initial-state", state_prep_circ);
  }

  auto adapt_params = config_params;
  adapt_params.merge(extra_params);
  auto adapt = xacc::getAlgorithm("adapt", adapt_params);
  auto buffer = xacc::qalloc(model.observable->nBits());
  adapt->execute(buffer);
  auto opt_val = (*buffer)["opt-val"].as<double>();
  auto opt_params = (*buffer)["opt-params"].as<std::vector<double>>();
  auto opt_ansatz = (*buffer)["opt-ansatz"].as<std::vector<int>>();
  auto final_ansatz_name = (*buffer)["final-ansatz"].as<std::string>();
  auto finalCircuit = xacc::getCompiled(final_ansatz_name);

  return {
      {"opt-val", opt_val},
      // Alias opt-val to energy as well
      {"energy", opt_val},
      {"opt-params", opt_params},
      {"opt-ansatz", opt_ansatz},
      {"circuit", finalCircuit}
  };
}
} // namespace QuaSiMo
} // namespace qcor