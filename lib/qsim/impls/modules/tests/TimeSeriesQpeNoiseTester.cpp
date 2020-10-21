#include "qcor.hpp"
#include "qcor_qsim.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"
#include <gtest/gtest.h>

// These tests take quite some time, hence just run one at a time...
//#define TEST_SIMPLE
//#define TEST_DEUTERON
#define TEST_ISING

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
#endif

#ifdef TEST_ISING
TEST(TimeSeriesQpeNoiseTester, checkIsingModel) {
  const int nbShots = 1024 * 1024;
  auto accelerator = xacc::getAccelerator("aer", {{"shots", nbShots}});
  using namespace qcor;
  const int nbQubits = 4;
  const double Jz = 1.0;
  const double Jx = 1.0;
  xacc::quantum::PauliOperator hamOpZ, hamOpX;
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
  auto evaluator = qsim::getObjEvaluator(hamOp, "qpe", {{"verified", true}});
  auto provider = xacc::getIRProvider("quantum");
  // Run the test with shots:
  xacc::internal_compiler::qpu = accelerator;
  // Variational parameters: 2 values for 1 layer:
  const double pz = 0.123;
  const double px = 0.567;
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

  // std::cout << "Kernel:\n" << kernel->toString() << "\n";
  const auto expVal = evaluator->evaluate(kernel);
  std::cout << "Exp-val:\n" << expVal << "\n";
}
#endif

int main(int argc, char **argv) {
  xacc::Initialize();
  xacc::set_verbose(true);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
