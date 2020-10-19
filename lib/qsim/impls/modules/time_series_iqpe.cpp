#include "Instruction.hpp"
#include "InstructionIterator.hpp"
#include "qsim_impl.hpp"
#include "utils/prony_method.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"

/// Implements time-series phase estimation protocol:
/// Refs:
/// https://arxiv.org/pdf/2010.02538.pdf

/// Notes:
/// We support both *unverified* and *verified* versions of the protocol.
/// i.e. for noise-free simulation, we can use the *unverified* version whereby
/// no post-selection based on measurement results of the main qubit register is
/// required.

namespace qcor {
namespace qsim {
double PhaseEstimationObjFuncEval::evaluate(
    std::shared_ptr<CompositeInstruction> state_prep) {
  // Default number of time steps to fit g(t)
  int nbSteps = 10;
  if (hyperParams.keyExists<int>("steps")) {
    nbSteps = hyperParams.get<int>("steps");
  }

  // If we run verify mode, (noisy simulator/hardware)
  bool verifyMode = false;
  if (hyperParams.keyExists<bool>("verified")) {
    verifyMode = hyperParams.get<bool>("verified");
  }

  const auto tList = xacc::linspace(0.0, 2 * M_PI, nbSteps);
  const auto nbQubits = target_operator->nBits();
  const size_t ancBit = nbQubits;

  auto provider = xacc::getIRProvider("quantum");
  auto state_prep_adjoint =
      provider->createComposite(state_prep->name() + "_adj");
  if (verifyMode) {
    std::vector<std::shared_ptr<xacc::Instruction>> gate_list;
    xacc::InstructionIterator it(state_prep);
    while (it.hasNext()) {
      auto nextInst = it.next();
      if (nextInst->isEnabled() && !nextInst->isComposite()) {
        gate_list.emplace_back(nextInst->clone());
      }
    }
    /// TODO: refactor this to be a common helper for QCOR
    /// to prevent code duplication.
    std::reverse(gate_list.begin(), gate_list.end());
    for (size_t i = 0; i < gate_list.size(); ++i) {
      auto &inst = gate_list[i];
      if (inst->name() == "Rx" || inst->name() == "Ry" ||
          inst->name() == "Rz" || inst->name() == "CPHASE" ||
          inst->name() == "U1" || inst->name() == "CRZ") {
        inst->setParameter(0, -inst->getParameter(0).template as<double>());
      } else if (inst->name() == "T") {
        auto tdg = provider->createInstruction("Tdg", inst->bits());
        std::swap(inst, tdg);
      } else if (inst->name() == "S") {
        auto sdg = provider->createInstruction("Sdg", inst->bits());
        std::swap(inst, sdg);
      }
    }
    state_prep_adjoint->addInstructions(gate_list);
  }

  std::vector<std::shared_ptr<CompositeInstruction>> fsToExec;
  // Time series data for a term: list of X and Y kernels (different times)
  using TimeSeriesData = std::vector<std::pair<std::string, std::string>>;
  using TermEvalData =
      std::vector<std::pair<std::complex<double>, TimeSeriesData>>;
  // Map from kernel name to result (expectation Z value)
  using ExecutionData = std::unordered_map<std::string, double>;

  TermEvalData obsTermTracking;
  for (auto &term : target_operator->getNonIdentitySubTerms()) {
    TimeSeriesData termData;
    // std::cout << "Evaluate: " << term->toString() << "\n";
    const auto termCoeff = term->coefficient();
    auto model = ModelBuilder::createModel(term.get());
    // High-level algorithm:
    // (I) For each time t:
    for (const auto &t : tList) {
      ///    (1) Apply the state_prep on the main qubit register
      static int count = 0;
      auto kernel = provider->createComposite("__TEMP__QPE__KERNEL__" +
                                              std::to_string(count++));
      kernel->addInstruction(state_prep);
      ///    (2) Estimate the <X> and <Y> for this time step
      auto qpeKernel =
          IterativeQpeWorkflow::constructQpeTrotterCircuit(term, t, nbQubits);
      kernel->addInstruction(qpeKernel);
      ///    (3) Add g(t) = <X> + i <Y>
      auto xKernel = provider->createComposite("__TEMP__QPE__KERNEL__X__" +
                                               std::to_string(count++));
      xKernel->addInstruction(kernel);
      xKernel->addInstruction(provider->createInstruction("H", ancBit));
      if (verifyMode) {
        // Add adjoint circuit.
        xKernel->addInstruction(state_prep_adjoint);
        // Measure the base register (for rejection sampling)
        for (size_t i = 0; i < nbQubits; ++i) {
          xKernel->addInstruction(provider->createInstruction("Measure", i));
        }
      }
      xKernel->addInstruction(provider->createInstruction("Measure", ancBit));

      auto yKernel = provider->createComposite("__TEMP__QPE__KERNEL__Y__" +
                                               std::to_string(count++));
      yKernel->addInstruction(kernel);
      yKernel->addInstruction(
          provider->createInstruction("Rx", {ancBit}, {M_PI_2}));
      if (verifyMode) {
        yKernel->addInstruction(state_prep_adjoint);
        for (size_t i = 0; i < nbQubits; ++i) {
          yKernel->addInstruction(provider->createInstruction("Measure", i));
        }
      }
      yKernel->addInstruction(provider->createInstruction("Measure", ancBit));

      fsToExec.emplace_back(xKernel);
      fsToExec.emplace_back(yKernel);
      termData.emplace_back(std::make_pair(xKernel->name(), yKernel->name()));
    }
    obsTermTracking.emplace_back(std::make_pair(termCoeff, termData));
  }

  auto qpu = xacc::internal_compiler::get_qpu();
  auto temp_buffer = xacc::qalloc(nbQubits + 1);
  // Execute all sub-kernels
  qpu->execute(temp_buffer, fsToExec);

  // Assemble execution data into a fast look-up map
  ExecutionData exeResult;

  /// TODO: handle rejection sampling if need verification/noise mitigation.
  // i.e. cannot rely on the default getExpectationValueZ but must manually
  // compute the expectation.
  for (auto &childBuffer : temp_buffer->getChildren()) {
    exeResult.emplace(childBuffer->name(), childBuffer->getExpectationValueZ());
  }

  std::complex<double> expVal =
      target_operator->getIdentitySubTerm()
          ? target_operator->getIdentitySubTerm()->coefficient()
          : 0.0;

  for (const auto &[coeff, listKernels] : obsTermTracking) {
    // g(t) function (for each term)
    std::vector<std::complex<double>> gFuncList;
    constexpr std::complex<double> I(0.0, 1.0);

    for (const auto &[xKernel, yKernel] : listKernels) {
      // Look-up the X and Y expectation for each time step.
      auto xIter = exeResult.find(xKernel);
      auto yIter = exeResult.find(yKernel);
      assert(xIter != exeResult.end());
      assert(yIter != exeResult.end());

      const double exp_x_val = xIter->second;
      const double exp_y_val = yIter->second;

      // Add the g(t) value from IQPE
      gFuncList.emplace_back(exp_x_val + I * exp_y_val);
    }
    assert(gFuncList.size() == tList.size());
    /// (II) Fit g(t) to determine A0 and A1
    /// DEBUG:
    // for (size_t i = 0; i < gFuncList.size(); ++i) {
    //   std::cout << "t = " << tList[i] << ": " << gFuncList[i] << "\n";
    // }
    auto pronyFit = qcor::utils::pronyFit(gFuncList);
    std::optional<double> A0, A1;
    std::optional<double> f0, f1;
    for (const auto &[ampl, phase] : pronyFit) {
      const double freq = std::arg(phase);
      const double amplitude = std::abs(ampl);
      // std::cout << "A = " << amplitude << "; "
      //           << "Freq = " << freq << "\n";
      constexpr double EPS = 1e-2;
      if (freq < 0 && std::abs(amplitude) > EPS) {
        assert(!A0.has_value());
        A0 = amplitude;
        f0 = freq;
      }
      if (freq > 0 && std::abs(amplitude) > EPS) {
        assert(!A1.has_value());
        A1 = amplitude;
        f1 = freq;
      }
    }
    assert(A0.has_value() || A1.has_value());
    if (A0.has_value() && A1.has_value()) {
      // Frequency values must be opposite.
      assert(std::abs(f0.value() + f1.value()) < 1e-6);
    }

    /// (III) Estimate <H> = (A0 - A1)/(A0 + A1)
    const double expValTerm = (A0.value_or(0.0) - A1.value_or(0.0)) /
                              (A0.value_or(0.0) + A1.value_or(0.0));
    expVal += (expValTerm * coeff);
  }

  return expVal.real();
}
} // namespace qsim
} // namespace qcor
