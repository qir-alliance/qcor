#include "qcor.hpp"
#include "qcor_qsim.hpp"
#include "utils/prony_method.hpp"
#include "xacc.hpp"
#include <gtest/gtest.h>

TEST(TimeSeriesQpeTester, checkPronyMethod) {
  auto x_vec = xacc::linspace(0.0, 1.0, 11);
  std::vector<std::complex<double>> y_vec;
  constexpr std::complex<double> I{0.0, 1.0};
  for (const auto &xVal : x_vec) {
    y_vec.emplace_back(
        0.5 * std::exp(I * xVal * 3.0) + 0.3 * std::exp(I * xVal * 5.0) +
        0.15 * std::exp(I * xVal * 1.5) + 0.1 * std::exp(I * xVal * 4.0) +
        0.05 * std::exp(I * xVal * 1.2));
  }
  auto result = qcor::utils::pronyFit(y_vec);
  size_t idx = 0;
  const std::vector<double> expectedAmpls{0.05, 0.1, 0.15, 0.3, 0.5};
  const std::vector<double> expectedFreqs{0.12, 0.4, 0.15, 0.5, 0.3};

  for (const auto &[ampl, phase] : result) {
    const auto freq = std::arg(phase);
    const auto amplitude = std::abs(ampl);
    EXPECT_NEAR(freq, expectedFreqs[idx], 1e-3);
    EXPECT_NEAR(amplitude, expectedAmpls[idx], 1e-3);
    std::cout << "A = " << amplitude << "; "
              << "Freq = " << freq << "\n";
    idx++;
  }
}

TEST(TimeSeriesQpeTester, checkSimple) {
  using namespace qcor;
  const auto angles = xacc::linspace(0.0, M_PI, 12);
  auto observable = Z(0);
  auto evaluator = qsim::getObjEvaluator(&observable, "qpe");
  auto provider = xacc::getIRProvider("quantum");
  xacc::internal_compiler::qpu = xacc::getAccelerator("qpp");

  for (const auto &angle : angles) {
    auto kernel = provider->createComposite("test");
    kernel->addInstruction(provider->createInstruction("Rx", {0}, {angle}));
    const auto expVal = evaluator->evaluate(kernel);
    const auto theoreticalExp = 1.0 - 2.0 * std::pow(std::sin(angle / 2.0), 2);
    std::cout << "Angle = " << angle << ": Exp val = " << expVal << " vs. "
              << theoreticalExp << "\n";
    EXPECT_NEAR(expVal, theoreticalExp, 1e-3);
  }
}

int main(int argc, char **argv) {
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
