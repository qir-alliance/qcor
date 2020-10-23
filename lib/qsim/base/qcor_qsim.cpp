#include "qcor_qsim.hpp"
#include "xacc_service.hpp"

namespace qcor {
namespace qsim {
bool CostFunctionEvaluator::initialize(Observable *observable,
                                       const HeterogeneousMap &params) {
  target_operator = observable;
  hyperParams = params;
  return target_operator != nullptr;
}

void executePassManager(
    std::vector<std::shared_ptr<CompositeInstruction>> evalKernels) {
  for (auto &subKernel : evalKernels) {
    execute_pass_manager(subKernel);
  }
}

QuantumSimulationModel
ModelBuilder::createModel(Observable *obs, TdObservable td_ham,
                          const HeterogeneousMap &params) {
  QuantumSimulationModel model;
  model.observable = obs;
  model.hamiltonian = td_ham;
  return model;
}

QuantumSimulationModel
ModelBuilder::createModel(Observable *obs, const HeterogeneousMap &params) {
  QuantumSimulationModel model;
  model.observable = obs;
  model.hamiltonian = [&](double t) {
    return *(static_cast<PauliOperator *>(obs));
  };
  return model;
}

QuantumSimulationModel
ModelBuilder::createModel(const std::string &format, const std::string &data,
                          const HeterogeneousMap &params) {
  QuantumSimulationModel model;
  // TODO:
  return model;
}

std::shared_ptr<QuantumSimulationWorkflow>
getWorkflow(const std::string &name, const HeterogeneousMap &init_params) {
  auto qsim_workflow = xacc::getService<QuantumSimulationWorkflow>(name);
  if (qsim_workflow && qsim_workflow->initialize(init_params)) {
    return qsim_workflow;
  }
  // ERROR: unknown workflow or invalid initialization options.
  return nullptr;
}

std::shared_ptr<CostFunctionEvaluator>
getObjEvaluator(Observable *observable, const std::string &name,
                const HeterogeneousMap &init_params) {
  auto evaluator = xacc::getService<CostFunctionEvaluator>(name);
  if (evaluator && evaluator->initialize(observable, init_params)) {
    return evaluator;
  }
  // ERROR: unknown CostFunctionEvaluator or invalid initialization options.
  return nullptr;
}
} // namespace qsim
} // namespace qcor