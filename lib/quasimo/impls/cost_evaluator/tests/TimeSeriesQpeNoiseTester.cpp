#include "qcor.hpp"
#include "qcor_qsim.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"
#include <gtest/gtest.h>

// These tests take quite some time, hence just run one at a time...
// Default for CI: only run the simple test
#define TEST_SIMPLE
//#define TEST_DEUTERON
//#define TEST_ISING

#ifdef TEST_SIMPLE
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
      QuaSiMo::getObjEvaluator(&observable, "qpe", {{"verified", true}});
  auto provider = xacc::getIRProvider("quantum");
  // Run the test with shots:
  xacc::internal_compiler::qpu = accelerator;

  for (const auto &angle : angles) {
    auto kernel = std::make_shared<qcor::CompositeInstruction>(
        provider->createComposite("test"));
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
#endif

#ifdef TEST_DEUTERON
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
      QuaSiMo::getObjEvaluator(&observable, "qpe", {{"verified", true}});
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
#endif

#ifdef TEST_ISING
TEST(TimeSeriesQpeNoiseTester, checkIsingModel) {
  const int nbShots = 1024 * 1024;
  // Error rate settings (for looking up noise model JSON file)
  const std::vector<std::string> errorRates = {"1e-3", "5e-3", "1e-2"};
  // Number of eval runs (draw random parameters for the ansatz)
  const size_t numTests = 50;
  // Result (statistics) for each noise setting:
  // VQPE, conventional partial tomography, and reference (true) result
  struct Result {
    std::vector<double> vqpe;
    std::vector<double> tomo;
    std::vector<double> ref;
  };

  std::unordered_map<std::string, Result> allResults;
  using namespace qcor;
  const int nbQubits = 4;
  const double Jz = 1.0;
  const double Jx = 1.0;
  xacc::quantum::Operator hamOpZ, hamOpX;
  for (int j = 0; j < nbQubits; ++j) {
    hamOpZ += (Jz * Z(j));
  }
  for (int j = 0; j < nbQubits; ++j) {
    // Periodic boundary condition (modulo nbQubits)
    hamOpX += (Jx * X(j) * X((j + 1) % nbQubits));
  }
  // Total Hamiltonian
  auto hamOp = hamOpZ + hamOpX;
  std::cout << "Ham:\n" << hamOp.toString() << "\n";
  // Run the QPE with verification.
  auto qpe_evaluator =
      QuaSiMo::getObjEvaluator(hamOp, "qpe", {{"verified", true}});
  auto tomo_evaluator = QuaSiMo::getObjEvaluator(hamOp);

  for (const auto &errorRate : errorRates) {
    std::cout << "Error rate = " << errorRate << "\n";
    Result resultForErrorRate;

    /// NOTE: this is a *synthetic* noise model which has a constant
    /// depolarizing noise moment on all gates.
    const std::string NOISE_MODEL_JSON_FILE = std::string(RESOURCE_DIR) +
                                              "/ibm_depol_" + errorRate +
                                              "_noise_model.json";
    std::ifstream noiseModelFile;
    noiseModelFile.open(NOISE_MODEL_JSON_FILE);
    std::stringstream noiseStrStream;
    noiseStrStream << noiseModelFile.rdbuf();
    const std::string noiseJsonStr = noiseStrStream.str();
    auto ref_accelerator = xacc::getAccelerator("qpp");
    auto noisy_accelerator = xacc::getAccelerator(
        "aer", {{"noise-model", noiseJsonStr}, {"shots", nbShots}});

    auto provider = xacc::getIRProvider("quantum");

    auto random_vector = [](const double l_range, const double r_range,
                            const std::size_t size) {
      // Generate a random initial parameter set
      std::random_device rnd_device;
      std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
      std::uniform_real_distribution<double> dist{l_range, r_range};
      auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
      std::vector<double> vec(size);
      std::generate(vec.begin(), vec.end(), gen);
      return vec;
    };

    // Draw random pz, px:
    const auto pz_vec = random_vector(-M_PI, M_PI, numTests);
    const auto px_vec = random_vector(-M_PI, M_PI, numTests);

    for (size_t testCaseId = 0; testCaseId < numTests; ++testCaseId) {
      // Variational parameters: 2 values for 1 layer:
      const double pz = pz_vec[testCaseId];
      const double px = px_vec[testCaseId];
      std::cout << "Px = " << px << "; Pz = " << pz << "\n";

      // Using only 1 layer (Eq. 73, p = 1)
      auto kernel = provider->createComposite("test");
      {
        auto expZ = std::dynamic_pointer_cast<xacc::quantum::Circuit>(
            xacc::getService<xacc::Instruction>("exp_i_theta"));
        EXPECT_TRUE(expZ->expand({std::make_pair("pauli", hamOpZ.toString())}));
        kernel->addInstruction(expZ->operator()({pz}));
      }
      {
        auto expX = std::dynamic_pointer_cast<xacc::quantum::Circuit>(
            xacc::getService<xacc::Instruction>("exp_i_theta"));
        EXPECT_TRUE(expX->expand({std::make_pair("pauli", hamOpX.toString())}));
        kernel->addInstruction(expX->operator()({px}));
      }
      // Compute the reference data:
      xacc::internal_compiler::qpu = ref_accelerator;
      // For extra validation, we make sure that both evaluators return the same
      // answer when there is no noise.
      const auto refExpVal1 = tomo_evaluator->evaluate(kernel);
      const auto refExpVal2 = qpe_evaluator->evaluate(kernel);
      EXPECT_NEAR(refExpVal1, refExpVal2, 1e-3);
      resultForErrorRate.ref.emplace_back(refExpVal1);

      // Run noisy evaluation:
      xacc::internal_compiler::qpu = noisy_accelerator;
      const auto tomoExpValNoise = tomo_evaluator->evaluate(kernel);
      const auto vqpeExpValNoise = qpe_evaluator->evaluate(kernel);
      resultForErrorRate.tomo.emplace_back(tomoExpValNoise);
      resultForErrorRate.vqpe.emplace_back(vqpeExpValNoise);
      std::cout << "Ref = " << refExpVal1 << "; Tomo = " << tomoExpValNoise
                << "; VQPE = " << vqpeExpValNoise << "\n";
    }

    EXPECT_EQ(resultForErrorRate.ref.size(), numTests);
    EXPECT_EQ(resultForErrorRate.tomo.size(), numTests);
    EXPECT_EQ(resultForErrorRate.vqpe.size(), numTests);

    allResults.emplace(errorRate, resultForErrorRate);
    // Save to CSV
    std::ofstream outFile;
    outFile.open("result.csv");
    outFile << "Error Rate, Ref, Tomo, VQPE\n";
    for (const auto &[errorRate, result] : allResults) {
      for (size_t i = 0; i < numTests; ++i) {
        outFile << errorRate << ", " << result.ref[i] << ", " << result.tomo[i]
                << ", " << result.vqpe[i] << "\n";
      }
    }
    outFile.close();
  }
}
#endif

int main(int argc, char **argv) {
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
