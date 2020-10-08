#include "qsim_impl.hpp"
#include "xacc_service.hpp"
#include "xacc.hpp"

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
    auto vqe = xacc::getAlgorithm("vqe");
    auto qpu = xacc::internal_compiler::get_qpu();

    OptFunction f(
        [&](const std::vector<double> &x, std::vector<double> &dx) {
          auto kernel = model.user_defined_ansatz->evaluate_kernel(x);
          // std::cout << "Kernel:\n" << kernel->toString() << "\n";
          auto success = vqe->initialize({{"ansatz", kernel},
                                          {"accelerator", qpu},
                                          {"observable", model.observable}});
          if (!success) {
            xacc::error("QCOR VQE Workflow Error - could not initialize "
                        "internal xacc vqe algorithm.");
          }
          auto tmp_child = qalloc(model.user_defined_ansatz->getQreg().size());
          auto energy =
              vqe->execute(xacc::as_shared_ptr(tmp_child.results()), {})[0];
          return energy;
        },
        nParams);

    auto result = optimizer->optimize(f);
    std::cout << "Min energy = " << result.first << "\n";
    return {{"energy", result.first}, {"opt-params", result.second}};
  }

  // TODO: support ansatz generation methods:
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
    context.RegisterService<qcor::QuatumSimulationWorkflow>(
        std::make_shared<qcor::TimeDependentWorkflow>());
    context.RegisterService<qcor::QuatumSimulationWorkflow>(
        std::make_shared<qcor::VqeWorkflow>());
  }

  void Stop(BundleContext) {}
};
} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(QuatumSimulationActivator)