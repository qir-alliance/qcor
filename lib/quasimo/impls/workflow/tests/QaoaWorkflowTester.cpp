#include "qcor.hpp"
#include "qcor_qsim.hpp"
#include "xacc.hpp"
#include <gtest/gtest.h>

TEST(QaoaWorkflowTester, checkGradientFree) {
  using namespace qcor;
  auto observable = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) +
                    .21829 * Z(0) - 6.125 * Z(1);
  xacc::internal_compiler::qpu = xacc::getAccelerator("qsim");
  auto problemModel = QuaSiMo::ModelFactory::createModel(&observable);
  auto optimizer = createOptimizer("nlopt", {{"nlopt-maxeval", 500}});
  auto workflow = QuaSiMo::getWorkflow("qaoa", {{"optimizer", optimizer}, {"steps", 8}});
  auto result = workflow->execute(problemModel);
  const auto energy = result.get<double>("energy");
  std::cout << "Min energy: " << energy << "\n";
  // EXPECT_NEAR(energy, -1.74, 0.25);
}

TEST(QaoaWorkflowTester, checkGradient) {
  using namespace qcor;
  auto observable = Z(0)*Z(1) + Z(0)*Z(2) + Z(1)*Z(2);
  xacc::internal_compiler::qpu = xacc::getAccelerator("qpp");
  auto problemModel = QuaSiMo::ModelFactory::createModel(&observable);
  auto optimizer = createOptimizer("mlpack");
  auto workflow = QuaSiMo::getWorkflow("qaoa", {{"optimizer", optimizer}});
  auto result = workflow->execute(problemModel);
  const auto energy = result.get<double>("energy");
  std::cout << "Min energy: " << energy << "\n";
  const double maxCutVal = -0.5*energy + 0.5*3;
  EXPECT_NEAR(maxCutVal, 2.0, 0.1);
}

int main(int argc, char **argv) {
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}