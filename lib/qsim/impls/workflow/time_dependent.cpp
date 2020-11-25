#include "time_dependent.hpp"
#include "qsim_utils.hpp"
#include "xacc_service.hpp"

namespace qcor {
namespace qsim {
bool TimeDependentWorkflow::initialize(const HeterogeneousMap &params) {
  // Get workflow parameters (specific to TD workflow):
  t_0 = 0.0;
  dt = 0.1;
  if (params.keyExists<double>("dt")) {
    dt = params.get<double>("dt");
  }
  int nbSteps = 1;
  if (params.keyExists<int>("steps")) {
    nbSteps = params.get<int>("steps");
  }

  t_final = nbSteps * dt;
  config_params = params;
  return true;
}

QuantumSimulationResult
TimeDependentWorkflow::execute(const QuantumSimulationModel &model) {
  QuantumSimulationResult result;
  evaluator = getEvaluator(model.observable, config_params);
  auto ham_func = model.hamiltonian;
  // A TD workflow: stepping through Trotter steps,
  // compute expectations at each step.
  double currentTime = t_0;
  std::vector<double> resultExpectationValues;
  std::shared_ptr<CompositeInstruction> totalCirc;
  // Just support Trotter for now
  // TODO: support different methods:
  auto method = xacc::getService<AnsatzGenerator>("trotter");
  for (;;) {
    // Evaluate the time-dependent Hamiltonian:
    auto ham_t = ham_func(currentTime);
    auto stepAnsatz = method->create_ansatz(&ham_t, {{"dt", dt}});
    // std::cout << "t = " << currentTime << "\n";
    // std::cout << stepAnsatz.circuit->toString() << "\n";
    // First step:
    if (!totalCirc) {
      totalCirc = stepAnsatz.circuit;
    } else {
      // Append Trotter steps
      totalCirc->addInstructions(stepAnsatz.circuit->getInstructions());
    }
    // std::cout << totalCirc->toString() << "\n";
    // Evaluate the expectation after these Trotter steps:
    const double ham_expect = evaluator->evaluate(totalCirc);
    resultExpectationValues.emplace_back(ham_expect);

    currentTime += dt;
    if (currentTime > t_final) {
      break;
    }
  }

  result.insert("exp-vals", resultExpectationValues);
  return result;
}
} // namespace qsim
} // namespace qcor