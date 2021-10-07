/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "qite.hpp"

#include "qsim_utils.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"

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
}  // namespace

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

  if (params.keyExists<std::shared_ptr<xacc::IRTransformation>>(
          "circuit-optimizer")) {
    extra_circuit_optimizers.push_back(
        params.get<std::shared_ptr<xacc::IRTransformation>>(
            "circuit-optimizer"));
  } else if (params.keyExists<
                 std::vector<std::shared_ptr<xacc::IRTransformation>>>(
                 "circuit-optimizers")) {
    auto opts =
        params.get<std::vector<std::shared_ptr<xacc::IRTransformation>>>(
            "circuit-optimizers");
    extra_circuit_optimizers.insert(extra_circuit_optimizers.end(),
                                    opts.begin(), opts.end());
  }

  config_params = params;
  return initializeOk;
}

QuantumSimulationResult QiteWorkflow::execute(
    const QuantumSimulationModel &model) {
  const auto nbSteps = config_params.get<int>("steps");
  const auto stepSize = config_params.get<double>("step-size");
  auto qite = xacc::getService<xacc::Algorithm>("qite");
  auto observable = xacc::as_shared_ptr(model.observable);
  const auto nbQubits = model.observable->nBits();
  auto acc = xacc::internal_compiler::get_qpu();
  qite->initialize({{"accelerator", acc},
                    {"steps", nbSteps},
                    {"observable", std::dynamic_pointer_cast<xacc::Observable>(
                                       observable->get_as_opaque())},
                    {"step-size", stepSize}});

  // Approximate imaginary-time Hamiltonian
  std::vector<Operator> approxOps;
  std::vector<double> energyAtStep;

  auto constructPropagateCircuit =
      [this, &acc](std::vector<Operator> &in_Aops,
             const std::shared_ptr<KernelFunctor> &in_statePrep,
             double in_stepSize) -> std::shared_ptr<CompositeInstruction> {
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
    for (auto &aObs : in_Aops) {
      // Circuit is: exp(-idt*A),
      // i.e. regular evolution which approximates the imaginary time evolution.
      for (auto &term : aObs.getNonIdentitySubTerms()) {
        auto method = xacc::getService<AnsatzGenerator>("trotter");
        auto trotterCir =
            method->create_ansatz(&term, {{"dt", 0.5 * in_stepSize}})
                .circuit;
        propagateKernel->addInstructions(trotterCir->getInstructions());
      }
    }

    for (auto &opt : extra_circuit_optimizers) {
      opt->apply(propagateKernel, acc, config_params);
    }

    // std::cout << "Progagated kernel:\n" << propagateKernel->toString() <<
    // "\n";
    return std::make_shared<CompositeInstruction>(propagateKernel);
  };

  // Cost function (observable) evaluator
  auto calcCurrentEnergy = [&]() {
    // Trotter kernel up to this point
    auto propagateKernel = constructPropagateCircuit(
        approxOps, model.user_defined_ansatz, stepSize);
    evaluator = getEvaluator(model.observable, config_params);
    return evaluator->evaluate(propagateKernel);
  };

  // Initial energy
  energyAtStep.emplace_back(calcCurrentEnergy());
  const auto pauliOps = generatePauliPermutation(nbQubits);
  const auto evaluateTomographyAtStep =
      [&](const std::shared_ptr<CompositeInstruction> &in_kernel) {
        // Observe the kernels using the various Pauli
        // operators to calculate S and b.
        std::vector<double> sigmaExpectation(pauliOps.size());
        sigmaExpectation[0] = 1.0;
        for (int i = 1; i < pauliOps.size(); ++i) {
          const std::string pauliObsStr = "1.0 " + pauliOps[i];
          std::shared_ptr<Operator> tomoObservable =
              std::make_shared<Operator>("pauli", pauliObsStr);
          assert(tomoObservable->getSubTerms().size() == 1);
          assert(tomoObservable->getNonIdentitySubTerms().size() == 1);
          auto temp_evaluator =
              getEvaluator(tomoObservable.get(), config_params);
          sigmaExpectation[i] = temp_evaluator->evaluate(in_kernel);
        }
        return sigmaExpectation;
      };

  // Main QITE time-stepping loop:
  for (int i = 0; i < nbSteps; ++i) {
    // Propagates the state via Trotter steps:
    auto kernel = constructPropagateCircuit(
        approxOps, model.user_defined_ansatz, stepSize);
    const std::vector<double> sigmaExpectation =
        evaluateTomographyAtStep(kernel);
    auto tmp_buffer = qalloc(nbQubits);
    auto normVal = qite->calculate(
        "approximate-ops", xacc::as_shared_ptr(tmp_buffer.results()),
        {{"pauli-ops", pauliOps}, {"pauli-ops-exp-val", sigmaExpectation}});
    const std::string AopsStr =
        tmp_buffer.results()->getInformation("Aops-str").as<std::string>();
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
          {"circuit", finalCircuit->as_xacc()}};
}
}  // namespace QuaSiMo
}  // namespace qcor