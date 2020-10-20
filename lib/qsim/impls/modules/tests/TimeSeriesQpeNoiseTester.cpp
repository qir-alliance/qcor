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
  auto accelerator = xacc::getAccelerator(
      "aer", {{"noise-model", noiseJsonStr}, {"shots", 8192}});

  using namespace qcor;
  const auto angles = xacc::linspace(0.0, M_PI, 12);
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
  }
}

int main(int argc, char **argv) {
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
