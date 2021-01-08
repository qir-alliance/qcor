#include "qcor.hpp"
#include "qcor_qsim.hpp"
#include "utils/qsim_utils.hpp"
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
  auto result = qcor::QuaSiMo::pronyFit(y_vec);
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
  const auto angles = xacc::linspace(0.0, M_PI, 8);
  auto observable = Z(0);
  auto evaluator = QuaSiMo::getObjEvaluator(&observable, "qpe");
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

TEST(TimeSeriesQpeTester, checkMultipleTerms) {
  using namespace qcor;
  const auto angles = xacc::linspace(0.0, M_PI, 3);
  auto observable = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) +
                    .21829 * Z(0) - 6.125 * Z(1);
  auto evaluator = QuaSiMo::getObjEvaluator(&observable, "qpe");
  // Reference evaluator (default tomography-based method)
  auto refEvaluator = QuaSiMo::getObjEvaluator(&observable);
  auto provider = xacc::getIRProvider("quantum");
  xacc::internal_compiler::qpu = xacc::getAccelerator("qpp");

  for (const auto &angle : angles) {
    auto kernel = provider->createComposite("test");
    kernel->addInstruction(provider->createInstruction("X", {0}));
    kernel->addInstruction(provider->createInstruction("Ry", {0}, {angle}));
    kernel->addInstruction(provider->createInstruction("CNOT", {1, 0}));
    const auto expVal = evaluator->evaluate(kernel);
    const auto refResult = refEvaluator->evaluate(kernel);
    std::cout << "Angle = " << angle << ": Exp val = " << expVal << " vs "
              << refResult << "\n";
    EXPECT_NEAR(expVal, refResult, 0.01);
  }
}

TEST(TimeSeriesQpeTester, checkVerifiedProtocolNoiseless) {
  using namespace qcor;
  const auto angles = xacc::linspace(0.0, M_PI, 3);
  auto observable = Z(0);
  // Run the QPE with verification.
  auto evaluator =
      QuaSiMo::getObjEvaluator(&observable, "qpe", {{"verified", true}});
  auto provider = xacc::getIRProvider("quantum");
  // Run the test with shots:
  xacc::internal_compiler::qpu =
      xacc::getAccelerator("aer", {{"shots", 10000}});

  for (const auto &angle : angles) {
    auto kernel = provider->createComposite("test");
    kernel->addInstruction(provider->createInstruction("Rx", {0}, {angle}));
    const auto expVal = evaluator->evaluate(kernel);
    const auto theoreticalExp = 1.0 - 2.0 * std::pow(std::sin(angle / 2.0), 2);
    std::cout << "Angle = " << angle << ": Exp val = " << expVal << " vs. "
              << theoreticalExp << "\n";
    // Since there is sampling error, we need to relax the limit.
    EXPECT_NEAR(expVal, theoreticalExp, 0.1);
  }
}

int main(int argc, char **argv) {
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
