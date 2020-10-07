#include "qsim_impl.hpp"
#include "xacc_service.hpp"

namespace qcor {
Ansatz TrotterEvolution::create_ansatz(Observable *obs,
                                       const HeterogeneousMap &params) {
  Ansatz result;
  // This ansatz generator requires an observable.
  assert(obs != nullptr);
  double dt = 1.0;
  if (params.keyExists<double>("dt")) {
    dt = params.get<double>("dt");
  }

  // Just use exp_i_theta for now
  // TODO: formalize a standard library kernel for this.
  auto expCirc = std::dynamic_pointer_cast<xacc::quantum::Circuit>(
      xacc::getService<xacc::Instruction>("exp_i_theta"));
  expCirc->expand({{"pauli", obs->toString()}});
  result.circuit = expCirc->operator()({dt});
  result.nb_qubits = expCirc->nRequiredBits();

  return result;
}

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

QsimWorkflow *TimeDependentWorkflow::getInstance() {
  if (!instance) {
    instance = new TimeDependentWorkflow();
  }
  return instance;
}

QsimModel ModelBuilder::createModel(Observable *obs, TdObservable td_ham,
                                    const HeterogeneousMap &params) {
  QsimModel model;
  model.observable = obs;
  model.hamiltonian = td_ham;
  return model;
}

QsimModel ModelBuilder::createModel(Observable *obs,
                                    const HeterogeneousMap &params) {
  QsimModel model;
  model.observable = obs;
  model.hamiltonian = [&](double t) {
    return *(static_cast<PauliOperator *>(obs));
  };
  return model;
}

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
  return true;
}

QsimResult TimeDependentWorkflow::execute(const QsimModel &model) {
  QsimResult result;

  // TODO: support multiple evaluator
  evaluator = CostFunctionEvaluator::getInstance();
  evaluator->initialize(model.observable);
  auto ham_func = model.hamiltonian;
  // A TD workflow: stepping through Trotter steps,
  // compute expectations at each step.
  double currentTime = t_0;
  std::vector<double> resultExpectationValues;
  std::shared_ptr<CompositeInstruction> totalCirc;
  // Just support Trotter for now
  // TODO: support different methods:
  TrotterEvolution method;
  for (;;) {
    // Evaluate the time-dependent Hamiltonian:
    auto ham_t = ham_func(currentTime);
    auto stepAnsatz = method.create_ansatz(&ham_t, {{"dt", dt}});
    // First step:
    if (!totalCirc) {
      totalCirc = stepAnsatz.circuit;
    } else {
      // Append Trotter steps
      totalCirc->addInstructions(stepAnsatz.circuit->getInstructions());
    }

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

QsimWorkflow *getWorkflow(WorkFlow type, const HeterogeneousMap &init_params) {
  // == TEMP-CODE
  // TODO: set-up service registry for workflow
  auto qsim_workflow = TimeDependentWorkflow::getInstance();
  if (qsim_workflow->initialize(init_params)) {
    return qsim_workflow;
  }

  return nullptr;
}

} // namespace qcor