#include "qsim_impl.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"

namespace qcor {
namespace qsim {
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

QuatumSimulationResult
TimeDependentWorkflow::execute(const QuatumSimulationModel &model) {
  QuatumSimulationResult result;
  evaluator = getObjEvaluator(model.observable);
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

bool VqeWorkflow::initialize(const HeterogeneousMap &params) {
  const std::string DEFAULT_OPTIMIZER = "nlopt";
  optimizer.reset();
  if (params.pointerLikeExists<Optimizer>("optimizer")) {
    optimizer =
        xacc::as_shared_ptr(params.getPointerLike<Optimizer>("optimizer"));
  } else {
    optimizer = createOptimizer(DEFAULT_OPTIMIZER);
  }
  // VQE workflow requires an optimizer
  return (optimizer != nullptr);
}

QuatumSimulationResult
VqeWorkflow::execute(const QuatumSimulationModel &model) {
  // If the model includes a concrete variational ansatz:
  if (model.user_defined_ansatz) {
    auto nParams = model.user_defined_ansatz->nParams();
    evaluator = getObjEvaluator(model.observable);
    auto qpu = xacc::internal_compiler::get_qpu();

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

bool IterativeQpeWorkflow::initialize(const HeterogeneousMap &params) {
  // Default params:
  num_steps = 1;
  num_iters = 1;
  if (params.keyExists<int>("time-steps")) {
    num_steps = params.get<int>("time-steps");
  }

  if (params.keyExists<int>("iterations")) {
    num_iters = params.get<int>("iterations");
  }

  return (num_steps >= 1) && (num_iters >= 1);
}

QuatumSimulationResult
IterativeQpeWorkflow::execute(const QuatumSimulationModel &model) {
  // 

  return {};
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

double
DefaultObjFuncEval::evaluate(std::shared_ptr<CompositeInstruction> state_prep) {
  // Reuse existing VQE util to evaluate the expectation value:
  auto vqe = xacc::getAlgorithm("vqe");
  auto qpu = xacc::internal_compiler::get_qpu();
  vqe->initialize({{"ansatz", state_prep},
                   {"accelerator", qpu},
                   {"observable", target_operator}});
  auto tmp_child = qalloc(state_prep->nPhysicalBits());
  auto energy = vqe->execute(xacc::as_shared_ptr(tmp_child.results()), {})[0];
  return energy;
}
} // namespace qsim
} // namespace qcor

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
namespace {
using namespace cppmicroservices;
class US_ABI_LOCAL QuatumSimulationActivator : public BundleActivator {

public:
  QuatumSimulationActivator() {}

  void Start(BundleContext context) {
    using namespace qcor;
    context.RegisterService<qsim::QuatumSimulationWorkflow>(
        std::make_shared<qsim::TimeDependentWorkflow>());
    context.RegisterService<qsim::QuatumSimulationWorkflow>(
        std::make_shared<qsim::VqeWorkflow>());
    context.RegisterService<qsim::QuatumSimulationWorkflow>(
        std::make_shared<qsim::IterativeQpeWorkflow>());
    context.RegisterService<qsim::CostFunctionEvaluator>(
        std::make_shared<qsim::DefaultObjFuncEval>());
  }

  void Stop(BundleContext) {}
};
} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(QuatumSimulationActivator)