#include "qite.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"
#include "qsim_utils.hpp"

namespace {
// Helper to generate all permutation of Pauli observables:
// e.g.
// 1-qubit: I, X, Y, Z
// 2-qubit: II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
std::vector<std::string> generatePauliPermutation(int in_nbQubits) {
  assert(in_nbQubits > 0);
  const int nbPermutations = std::pow(4, in_nbQubits);
  std::vector<std::string> opsList;
  opsList.reserve(nbPermutations);

  const std::vector<std::string> pauliOps{"X", "Y", "Z"};
  const auto addQubitPauli = [&opsList, &pauliOps](int in_qubitIdx) {
    const auto currentOpListSize = opsList.size();
    for (int i = 0; i < currentOpListSize; ++i) {
      auto &currentOp = opsList[i];
      for (const auto &pauliOp : pauliOps) {
        const auto newOp = currentOp + pauliOp + std::to_string(in_qubitIdx);
        opsList.emplace_back(newOp);
      }
    }
  };

  opsList = {"", "X0", "Y0", "Z0"};
  for (int i = 1; i < in_nbQubits; ++i) {
    addQubitPauli(i);
  }

  assert(opsList.size() == nbPermutations);
  std::sort(opsList.begin(), opsList.end());

  return opsList;
};
} // namespace

namespace qcor {
namespace QuaSiMo {
bool QiteWorkflow::initialize(const HeterogeneousMap &params) {
  bool initializeOk = true;
  if (!params.keyExists<int>("steps")) {
    std::cout << "'steps' is required.\n";
    initializeOk = false;
  }

  if (!params.keyExists<double>("step-size")) {
    std::cout << "'step-size' is required.\n";
    initializeOk = false;
  }
  config_params = params;
  return initializeOk;
}

QuantumSimulationResult
QiteWorkflow::execute(const QuantumSimulationModel &model) {
  const auto nbSteps = config_params.get<int>("steps");
  const auto stepSize = config_params.get<double>("step-size");
  auto qite = xacc::getService<xacc::Algorithm>("qite");
  auto observable = xacc::as_shared_ptr(model.observable);
  const auto nbQubits = model.observable->nBits();
  auto acc = xacc::internal_compiler::get_qpu();
  qite->initialize({{"accelerator", acc},
                    {"steps", nbSteps},
                    {"observable", observable},
                    {"step-size", stepSize}});
  
  // Approximate imaginary-time Hamiltonian
  std::vector<std::shared_ptr<Observable>> approxOps;
  std::vector<double> energyAtStep;

  auto constructPropagateCircuit =
      [](const std::vector<std::shared_ptr<Observable>> &in_Aops,
         const std::shared_ptr<KernelFunctor> &in_statePrep, double in_stepSize)
      -> std::shared_ptr<CompositeInstruction> {
    auto gateRegistry = xacc::getService<xacc::IRProvider>("quantum");
    auto propagateKernel = gateRegistry->createComposite("statePropCircuit");

    // Adds ansatz if provided
    if (in_statePrep) {
      auto statePrep = in_statePrep->evaluate_kernel({});
      propagateKernel->addInstructions(statePrep->getInstructions());
    }

    // Progagates by Trotter steps
    // Using those A operators that have been
    // optimized up to this point.
    for (const auto &aObs : in_Aops) {
      // Circuit is: exp(-idt*A),
      // i.e. regular evolution which approximates the imaginary time evolution.
      for (const auto &term : aObs->getNonIdentitySubTerms()) {
        auto method = xacc::getService<AnsatzGenerator>("trotter");
        auto trotterCir = method->create_ansatz(term.get(), {{"dt", 0.5 * in_stepSize}}).circuit;
        propagateKernel->addInstructions(trotterCir->getInstructions());
      }
    }

    // std::cout << "Progagated kernel:\n" << propagateKernel->toString() <<
    // "\n";
    return propagateKernel;
  };
  
  // Cost function (observable) evaluator
  auto calcCurrentEnergy = [&](){
    // Trotter kernel up to this point
    auto propagateKernel = constructPropagateCircuit(approxOps, model.user_defined_ansatz, stepSize);
    evaluator = getEvaluator(model.observable, config_params);
    return evaluator->evaluate(propagateKernel);
  };

  // Initial energy
  energyAtStep.emplace_back(calcCurrentEnergy());
  const auto pauliOps = generatePauliPermutation(nbQubits);
  const auto evaluateTomographyAtStep = [&](const std::shared_ptr<CompositeInstruction>& in_kernel) {
    // Observe the kernels using the various Pauli
    // operators to calculate S and b.
    std::vector<double> sigmaExpectation(pauliOps.size());
    sigmaExpectation[0] = 1.0;
    for (int i = 1; i < pauliOps.size(); ++i) {
      std::shared_ptr<Observable> tomoObservable =
          std::make_shared<xacc::quantum::PauliOperator>();
      const std::string pauliObsStr = "1.0 " + pauliOps[i];
      tomoObservable->fromString(pauliObsStr);
      assert(tomoObservable->getSubTerms().size() == 1);
      assert(tomoObservable->getNonIdentitySubTerms().size() == 1);
      auto temp_evaluator = getEvaluator(tomoObservable.get(), config_params);
      sigmaExpectation[i] = temp_evaluator->evaluate(in_kernel);
    }
    return sigmaExpectation;
  };

  // Main QITE time-stepping loop:
  for (int i = 0; i < nbSteps; ++i) {
    // Propagates the state via Trotter steps:
    auto kernel = constructPropagateCircuit(approxOps, model.user_defined_ansatz, stepSize);
    const std::vector<double> sigmaExpectation = evaluateTomographyAtStep(kernel);
    auto tmp_buffer = qalloc(nbQubits);
    auto normVal = qite->calculate(
        "approximate-ops", xacc::as_shared_ptr(tmp_buffer.results()),
        {{"pauli-ops", pauliOps}, {"pauli-ops-exp-val", sigmaExpectation}});
    const std::string AopsStr = tmp_buffer.results()->getInformation("Aops-str").as<std::string>();
    auto new_approx_ops = createObservable(AopsStr);
    approxOps.emplace_back(new_approx_ops);
    energyAtStep.emplace_back(calcCurrentEnergy());
  }

  // Returns:
  // - Final energy
  // - Energy values for all QITE time steps.
  // - Final QITE circuit
  auto finalCircuit =
      constructPropagateCircuit(approxOps, model.user_defined_ansatz, stepSize);
  return {{"energy", energyAtStep.back()},
          {"exp-vals", energyAtStep},
          {"circuit", finalCircuit}};
}
} // namespace QuaSiMo
} // namespace qcor