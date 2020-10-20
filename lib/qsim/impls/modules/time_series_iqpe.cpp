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
  // Note: for Pauli operators, we expect that there are 2 eigenvalues for each
  // term, hence use a minimal number of steps, which is 5.
  int nbSteps = 5;
  if (hyperParams.keyExists<int>("steps")) {
    nbSteps = hyperParams.get<int>("steps");
  }

  // If we run verify mode, (noisy simulator/hardware)
  bool verifyMode = false;
  if (hyperParams.keyExists<bool>("verified")) {
    verifyMode = hyperParams.get<bool>("verified");
  }
  // Minimum: 2 freqs (eigenvalues) -> 5 data points.
  if (nbSteps < 5) {
    xacc::error("Not enough time-series data samples for frequency/eigenvalue "
                "estimation.");
    return 0.0;
  }

  const auto tList = xacc::linspace(0.0, 2 * M_PI, nbSteps);
  const double SAMPLING_FREQ = (nbSteps - 1) / (2 * M_PI);
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
    // Normalize the term so that the signal processing works as expected.
    auto pauliCast = std::static_pointer_cast<PauliOperator>(term);
    if (pauliCast) {
      pauliCast->operator*=(1.0 / termCoeff);
    } else {
      xacc::error("Only fast-forwardable operators (Pauli) are supported.");
      return 0.0;
    }

    // High-level algorithm:
    // (I) For each time t:
    for (const auto &t : tList) {
      ///    (1) Apply the state_prep on the main qubit register
      static int count = 0;
      auto kernel = provider->createComposite("__TEMP__QPE__KERNEL__" +
                                              std::to_string(count++));
      kernel->addInstruction(state_prep);
      ///    (2) Estimate the <X> and <Y> for this time step
      auto qpeKernel = IterativeQpeWorkflow::constructQpeTrotterCircuit(
          pauliCast, t, nbQubits);
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

      static bool printedOnce = false;
      if (!printedOnce) {
        xacc::info("X-basis kernel: \n" + xKernel->toString());
        xacc::info("Y-basis kernel: \n" + yKernel->toString());
        printedOnce = true;
      }
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

  /// Handle rejection sampling if need verification/noise mitigation.
  // i.e. cannot rely on the default getExpectationValueZ but must manually
  // compute the expectation.
  const auto mitigateMeasurementResult =
      [&](const std::map<std::string, int> &in_rawResult) {
        std::map<std::string, int> result{{"0", 0}, {"1", 0}};
        const size_t bitIdx =
            (qpu->getBitOrder() == Accelerator::BitOrder::MSB) ? 0 : nbQubits;
        const std::string CORRECT_VERIFIED_BITSTRING(nbQubits, '0');
        size_t totalCount = 0;
        for (const auto &[bitString, count] : in_rawResult) {
          totalCount += count;
          assert(bitString.size() == nbQubits + 1);
          const std::string bitVal = bitString.substr(bitIdx, 1);
          std::string verifiedBitString = bitString;
          verifiedBitString.erase(verifiedBitString.begin() + bitIdx);
          if (verifiedBitString == CORRECT_VERIFIED_BITSTRING) {
            result[bitVal] += count;
          }
        }

        return std::make_pair(totalCount, result);
      };

  // Compensation for spurious eigenvalues due to sampling noise.
  // See Appendix C (https://arxiv.org/pdf/2010.02538.pdf)
  double factor = 1.0;

  // Mitigate/verify the result if in the 'verified' mode:
  for (auto &childBuffer : temp_buffer->getChildren()) {
    if (verifyMode && !childBuffer->getMeasurementCounts().empty()) {
      auto [totalCount, mitigatedResult] =
          mitigateMeasurementResult(childBuffer->getMeasurementCounts());
      assert(mitigatedResult.size() == 2);
      factor = 1.0 / (1.0 + std::sqrt(nbSteps - 2) / std::sqrt(totalCount));
      const int m0_verified = mitigatedResult["0"];
      const int m1_verified = mitigatedResult["1"];
      const int totalVerified = m0_verified + m1_verified;
      if (totalVerified == 0) {
        xacc::error("Failed to mitigate QPE results: no valid bit string after "
                    "verification.");
        return 0.0;
      } else {
        xacc::info("Buffer '" + childBuffer->name() +
                   "': verified count = " + std::to_string(totalVerified));

        // See Eq. 4 (https://arxiv.org/pdf/2010.02538.pdf)
        // Note: the denominator is the total count, not the sum of verified
        // count, i.e. not strictly post-selection
        const double verifiedExp =
            static_cast<double>(m0_verified - m1_verified) / totalCount;
        // std::cout << "m0 = " << m0_verified << ", m1 = " << m1_verified
        //           << "; Exp = " << verifiedExp << "\n";
        exeResult.emplace(childBuffer->name(), verifiedExp);
      }
    } else {
      exeResult.emplace(childBuffer->name(),
                        childBuffer->getExpectationValueZ());
    }
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
    double expValTerm = 0.0;
    for (const auto &[ampl, phase] : pronyFit) {
      const double freq = -std::arg(phase) * SAMPLING_FREQ;
      const double amplitude = std::abs(ampl);
      // std::cout << "A = " << amplitude << "; "
      //           << "Freq = " << freq << "\n";
      // Generic expectation value estimation (Eq. 22)
      expValTerm += (freq * amplitude);
    }
    // Compensate for sampling noise (factor)
    expVal += (factor * expValTerm * coeff);
  }

  return expVal.real();
}
} // namespace qsim
} // namespace qcor
