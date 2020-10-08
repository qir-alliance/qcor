#include "qcor_qsim.hpp"
#include "xacc_service.hpp"

namespace qcor {
bool CostFunctionEvaluator::initialize(Observable *observable,
                                       const HeterogeneousMap &params) {
  target_operator = observable;
  // TODO: use qcor data
  quantum_backend = nullptr;
  if (params.pointerLikeExists<Accelerator>("accelerator")) {
    quantum_backend = params.getPointerLike<Accelerator>("accelerator");
  }

  hyperParams = params;
  return target_operator && quantum_backend;
}

CostFunctionEvaluator *CostFunctionEvaluator::getInstance() {
  if (!instance) {
    instance = new CostFunctionEvaluator();
  }
  return instance;
}

double CostFunctionEvaluator::evaluate(
    std::shared_ptr<CompositeInstruction> state_prep) {

  // Measure the observables:
  // TODO: Port the existing VQE impl. as the default.

  // TODO:
  return 0.0;
}

QuatumSimulationModel
ModelBuilder::createModel(Observable *obs, TdObservable td_ham,
                          const HeterogeneousMap &params) {
  QuatumSimulationModel model;
  model.observable = obs;
  model.hamiltonian = td_ham;
  return model;
}

QuatumSimulationModel
ModelBuilder::createModel(Observable *obs, const HeterogeneousMap &params) {
  QuatumSimulationModel model;
  model.observable = obs;
  model.hamiltonian = [&](double t) {
    return *(static_cast<PauliOperator *>(obs));
  };
  return model;
}

QuatumSimulationModel
ModelBuilder::createModel(const std::string &format, const std::string &data,
                          const HeterogeneousMap &params) {
  QuatumSimulationModel model;
  // TODO:
  return model;
}

std::shared_ptr<QuatumSimulationWorkflow>
getWorkflow(const std::string &name, const HeterogeneousMap &init_params) {
  auto qsim_workflow = xacc::getService<QuatumSimulationWorkflow>(name);
  if (qsim_workflow && qsim_workflow->initialize(init_params)) {
    return qsim_workflow;
  }
  // ERROR: unknown workflow or invalid initialization options.
  return nullptr;
}
} // namespace qcor