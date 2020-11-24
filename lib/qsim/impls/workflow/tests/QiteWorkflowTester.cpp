#include "qcor.hpp"
#include "qcor_qsim.hpp"
#include "xacc.hpp"
#include <gtest/gtest.h>

TEST(QiteWorkflowTester, checkSimple) {
  using namespace qcor;
  auto observable = 0.7071067811865475 * X(0) + 0.7071067811865475 * Z(0);
  xacc::internal_compiler::qpu = xacc::getAccelerator("qpp");
  const int nbSteps = 25;
  const double stepSize = 0.1;
  auto problemModel = qsim::ModelBuilder::createModel(&observable);
  auto workflow =
      qsim::getWorkflow("qite", {{"steps", nbSteps}, {"step-size", stepSize}});
  auto result = workflow->execute(problemModel);
  const auto energy = result.get<double>("energy");
  const auto energyAtStep = result.get<std::vector<double>>("exp-vals");
  for (const auto &val : energyAtStep) {
    std::cout << val << "\n";
  }
  // Minimal Energy = -1
  EXPECT_NEAR(energy, -1.0, 1e-2);
  auto finalCircuit = result.getPointerLike<xacc::CompositeInstruction>("circuit");
  std::cout << "HOWDY:\n" << finalCircuit->toString() << "\n";
}

int main(int argc, char **argv) {
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}