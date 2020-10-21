#include "qcor.hpp"
#include "qcor_qsim.hpp"
#include "xacc.hpp"
#include <gtest/gtest.h>

TEST(TimeSeriesQpeNoiseTester, checkSimple) {
  const std::string NOISE_MODEL_JSON_FILE =
      std::string(RESOURCE_DIR) + "/ibmqx2_noise_model.json";
  std::ifstream noiseModelFile;
  noiseModelFile.open(NOISE_MODEL_JSON_FILE);
  std::stringstream noiseStrStream;
  noiseStrStream << noiseModelFile.rdbuf();
  const std::string noiseJsonStr = noiseStrStream.str();
  // See FIG. 7.: https://arxiv.org/pdf/2010.02538.pdf
  // Number of samples required for good convergence: ~ 10^6 - 10^7
  const int nbShots = 1024 * 32;
  auto accelerator = xacc::getAccelerator(
      "aer", {{"noise-model", noiseJsonStr}, {"shots", nbShots}});

  using namespace qcor;
  const auto angles = xacc::linspace(M_PI / 4.0, 3.0 * M_PI / 4.0, 3);
  auto observable = Z(0);
  // Run the QPE with verification.
  auto evaluator =
      qsim::getObjEvaluator(&observable, "qpe", {{"verified", true}});
  auto provider = xacc::getIRProvider("quantum");
  // Run the test with shots:
  xacc::internal_compiler::qpu = accelerator;

  for (const auto &angle : angles) {
    auto kernel = provider->createComposite("test");
    kernel->addInstruction(provider->createInstruction("Rx", {0}, {angle}));
    const auto expVal = evaluator->evaluate(kernel);
    const auto theoreticalExp = 1.0 - 2.0 * std::pow(std::sin(angle / 2.0), 2);
    std::cout << "Angle = " << angle << ": Exp val = " << expVal << " vs. "
              << theoreticalExp << "\n";
    // Under noisy condition, the error is quite high: ~ 0.1 (10^-1).
    // see Fig. 7 of https://arxiv.org/pdf/2010.02538.pdf
    EXPECT_NEAR(expVal, theoreticalExp, 0.1);
  }
}

TEST(TimeSeriesQpeNoiseTester, checkDeuteron) {
  const std::string NOISE_MODEL_JSON_FILE =
      std::string(RESOURCE_DIR) + "/ibmqx2_noise_model.json";
  std::ifstream noiseModelFile;
  noiseModelFile.open(NOISE_MODEL_JSON_FILE);
  std::stringstream noiseStrStream;
  noiseStrStream << noiseModelFile.rdbuf();
  const std::string noiseJsonStr = noiseStrStream.str();
  // See FIG. 7.: https://arxiv.org/pdf/2010.02538.pdf
  // Number of samples required for good convergence: ~ 10^6 - 10^7
  const int nbShots = 1024 * 1024;
  auto accelerator = xacc::getAccelerator(
      "aer", {{"noise-model", noiseJsonStr}, {"shots", nbShots}});

  using namespace qcor;
  const auto angles = xacc::linspace(0.0, M_PI, 10);
  auto observable = 5.907 - 2.1433 * X(0) * X(1) - 2.143 * Y(0) * Y(1) +
                    0.21829 * Z(0) - 6.125 * Z(1);
  // Run the QPE with verification.
  auto evaluator =
      qsim::getObjEvaluator(&observable, "qpe", {{"verified", true}});
  auto provider = xacc::getIRProvider("quantum");
  // Run the test with shots:
  xacc::internal_compiler::qpu = accelerator;

  for (const auto &angle : angles) {
    auto kernel = provider->createComposite("test");
    kernel->addInstruction(provider->createInstruction("X", {0}));
    kernel->addInstruction(provider->createInstruction("Ry", {0}, {angle}));
    kernel->addInstruction(provider->createInstruction("CNOT", {1, 0}));
    const auto expVal = evaluator->evaluate(kernel);
    std::cout << "Angle = " << angle << ": Exp val = " << expVal << "\n";
  }
}

int main(int argc, char **argv) {
  xacc::Initialize();
  xacc::set_verbose(true);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
