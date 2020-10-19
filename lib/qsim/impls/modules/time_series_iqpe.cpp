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

  const auto tList = xacc::linspace(0.0, 2 * M_PI, nbSteps);
  std::complex<double> expVal =
      target_operator->getIdentitySubTerm()
          ? target_operator->getIdentitySubTerm()->coefficient()
          : 0.0;
  const auto nbQubits = target_operator->nBits();
  const size_t ancBit = nbQubits;
  // TODO: combine sub-circuits for all terms into a vector of
  // CompositeInstructions.
  for (auto &term : target_operator->getNonIdentitySubTerms()) {
    // std::cout << "Evaluate: " << term->toString() << "\n";
    const auto termCoeff = term->coefficient();
    auto model = ModelBuilder::createModel(term.get());

    // g(t) function
    std::vector<std::complex<double>> gFuncList;
    constexpr std::complex<double> I(0.0, 1.0);
    // High-level algorithm:
    // (I) For each time t:
    for (const auto &t : tList) {
      ///    (1) Apply the state_prep on the main qubit register
      auto provider = xacc::getIRProvider("quantum");
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
      xKernel->addInstruction(provider->createInstruction("Measure", ancBit));

      auto yKernel = provider->createComposite("__TEMP__QPE__KERNEL__Y__" +
                                               std::to_string(count++));
      yKernel->addInstruction(kernel);
      yKernel->addInstruction(
          provider->createInstruction("Rx", {ancBit}, {M_PI_2}));
      yKernel->addInstruction(provider->createInstruction("Measure", ancBit));

      // std::cout << "X Kernel:\n" << xKernel->toString() << "\n";
      // std::cout << "Y Kernel:\n" << yKernel->toString() << "\n";

      auto qpu = xacc::internal_compiler::get_qpu();
      auto temp_buffer1 = xacc::qalloc(nbQubits + 1);
      qpu->execute(temp_buffer1, xKernel);
      auto temp_buffer2 = xacc::qalloc(nbQubits + 1);
      qpu->execute(temp_buffer2, yKernel);
      auto exp_x_val = temp_buffer1->getExpectationValueZ();
      auto exp_y_val = temp_buffer2->getExpectationValueZ();

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
    expVal += (expValTerm * termCoeff);
  }

  return expVal.real();
}
} // namespace qsim
} // namespace qcor
