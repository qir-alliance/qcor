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

QuantumSimulationResult
TimeDependentWorkflow::execute(const QuantumSimulationModel &model) {
  QuantumSimulationResult result;
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

QuantumSimulationResult
VqeWorkflow::execute(const QuantumSimulationModel &model) {
  // If the model includes a concrete variational ansatz:
  if (model.user_defined_ansatz) {
    auto nParams = model.user_defined_ansatz->nParams();
    evaluator = getObjEvaluator(model.observable);

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

std::shared_ptr<CompositeInstruction>
IterativeQpeWorkflow::constructQpeTrotterCircuit(
    std::shared_ptr<Observable> obs, double trotter_step, int steps, int k,
    double omega) {
  auto provider = xacc::getIRProvider("quantum");
  auto kernel = provider->createComposite("__TEMP__QPE__LOOP__");
  const auto nbQubits = obs->nBits();
  // Ancilla qubit is the last one in the register.
  const size_t ancBit = nbQubits;

  // Hadamard on ancilla qubit
  kernel->addInstruction(provider->createInstruction("H", ancBit));

  // Using Trotter evolution method to generate U:
  // TODO: support other methods (e.g. Suzuki)
  TrotterEvolution method;
  auto trotterCir =
      method.create_ansatz(obs.get(), {{"dt", trotter_step}}).circuit;
  // std::cout << "Trotter circ:\n" << trotterCir->toString() << "\n";

  // Controlled-U
  auto ctrlKernel = std::dynamic_pointer_cast<CompositeInstruction>(
      xacc::getService<xacc::Instruction>("C-U"));
  ctrlKernel->expand({
      std::make_pair("U", trotterCir),
      std::make_pair("control-idx", static_cast<int>(ancBit)),
  });

  // Apply C-U^n
  int power = 1 << (k - 1);
  for (int i = 0; i < power * steps; ++i) {
    for (int instId = 0; instId < ctrlKernel->nInstructions(); ++instId) {
      // We need to clone the instruction since it'll be repeated.
      kernel->addInstruction(ctrlKernel->getInstruction(instId)->clone());
    }
  }

  // Rz on ancilla qubit
  // Global phase due to identity pauli
  if (obs->getIdentitySubTerm()) {
    const double idCoeff = obs->getIdentitySubTerm()->coefficient().real();
    const double globalPhase = 2 * M_PI * idCoeff * power;
    // std::cout << "Global phase = " << globalPhase << "\n";
    kernel->addInstruction(
        provider->createInstruction("Rz", {ancBit}, {globalPhase}));
  }

  kernel->addInstruction(provider->createInstruction("Rz", {ancBit}, {omega}));
  return kernel;
}

std::shared_ptr<CompositeInstruction> IterativeQpeWorkflow::constructQpeCircuit(
    std::shared_ptr<Observable> obs, int k, double omega, bool measure) const {
  auto provider = xacc::getIRProvider("quantum");
  const double trotterStepSize = -2 * M_PI / num_steps;
  auto kernel = constructQpeTrotterCircuit(obs, trotterStepSize, num_steps, k, omega);
  const auto nbQubits = obs->nBits();
  
  // Ancilla qubit is the last one in the register
  const size_t ancBit = nbQubits;
  // Hadamard on ancilla qubit (measure in X basis for regular IQPE)
  kernel->addInstruction(provider->createInstruction("H", ancBit));

  if (measure) {
    kernel->addInstruction(provider->createInstruction("Measure", ancBit));
  }

  return kernel;
}

void IterativeQpeWorkflow::HamOpConverter::fromObservable(Observable *obs) {
  translation = 0.0;
  for (auto &term : obs->getSubTerms()) {
    translation += std::abs(term->coefficient());
  }
  stretch = 0.5 / translation;
}

std::shared_ptr<Observable>
IterativeQpeWorkflow::HamOpConverter::stretchObservable(Observable *obs) const {
  PauliOperator *pauliCast = static_cast<PauliOperator *>(obs);
  if (pauliCast) {
    auto result = std::make_shared<PauliOperator>(translation);
    result->operator+=(*pauliCast);
    result->operator*=(stretch);
    return result;
  } else {
    return nullptr;
  }
}

double
IterativeQpeWorkflow::HamOpConverter::computeEnergy(double phaseVal) const {
  return phaseVal / stretch - translation;
}

QuantumSimulationResult
IterativeQpeWorkflow::execute(const QuantumSimulationModel &model) {
  ham_converter.fromObservable(model.observable);
  auto stretchedObs = ham_converter.stretchObservable(model.observable);
  // std::cout << "Stretched Obs: " << stretchedObs->toString() << "\n";
  auto provider = xacc::getIRProvider("quantum");
  // Iterative Quantum Phase Estimation:
  // We're using XACC IR construction API here, since using QCOR kernels here
  // seems to be complicated.
  double omega_coef = 0.0;
  // Iterates over the num_iters
  // k runs from the number of iterations back to 1
  for (int iterIdx = 0; iterIdx < num_iters; ++iterIdx) {
    // State prep: evolves the qubit register to the initial quantum state, i.e.
    // the eigenvector state to estimate the eigenvalue.
    auto kernel = provider->createComposite("__TEMP__ITER_QPE__");
    if (model.user_defined_ansatz) {
      kernel->addInstruction(model.user_defined_ansatz->evaluate_kernel({}));
    }
    omega_coef = omega_coef / 2.0;
    // Construct the QPE circuit and append to the kernel:
    auto k = num_iters - iterIdx;

    auto iterQpe = constructQpeCircuit(stretchedObs, k, -2 * M_PI * omega_coef);
    kernel->addInstruction(iterQpe);
    // Executes the iterative QPE algorithm:
    auto qpu = xacc::internal_compiler::get_qpu();
    auto temp_buffer = xacc::qalloc(stretchedObs->nBits() + 1);
    // std::cout << "Kernel: \n" << kernel->toString() << "\n";

    qpu->execute(temp_buffer, kernel);
    // temp_buffer->print();

    // Estimate the phase value's bit at this iteration,
    // i.e. get the most-probable measure bit.
    const bool bitResult = [&temp_buffer]() {
      if (!temp_buffer->getMeasurementCounts().empty()) {
        // If the QPU returns bitstrings:
        if (xacc::container::contains(temp_buffer->getMeasurements(), "0")) {
          if (xacc::container::contains(temp_buffer->getMeasurements(), "1")) {
            return temp_buffer->computeMeasurementProbability("1") >
                   temp_buffer->computeMeasurementProbability("0");
          } else {
            return false;
          }
        } else {
          assert(
              xacc::container::contains(temp_buffer->getMeasurements(), "1"));
          return true;
        }
      } else {
        // If the QPU returns *expected* Z value:
        return temp_buffer->getExpectationValueZ() < 0.0;
      }
    }();

    if (bitResult) {
      omega_coef = omega_coef + 0.5;
    }
    // std::cout << "Iter " << iterIdx << ": Result = " << bitResult << ";
    // omega_coef = " << omega_coef << "\n";
  }

  return {{"phase", omega_coef},
          {"energy", ham_converter.computeEnergy(omega_coef)}};
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
class US_ABI_LOCAL QuantumSimulationActivator : public BundleActivator {

public:
  QuantumSimulationActivator() {}

  void Start(BundleContext context) {
    using namespace qcor;
    context.RegisterService<qsim::QuantumSimulationWorkflow>(
        std::make_shared<qsim::TimeDependentWorkflow>());
    context.RegisterService<qsim::QuantumSimulationWorkflow>(
        std::make_shared<qsim::VqeWorkflow>());
    context.RegisterService<qsim::QuantumSimulationWorkflow>(
        std::make_shared<qsim::IterativeQpeWorkflow>());
    context.RegisterService<qsim::CostFunctionEvaluator>(
        std::make_shared<qsim::DefaultObjFuncEval>());
    context.RegisterService<qsim::CostFunctionEvaluator>(
        std::make_shared<qsim::PhaseEstimationObjFuncEval>());
  }

  void Stop(BundleContext) {}
};
} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(QuantumSimulationActivator)