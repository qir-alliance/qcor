/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include <gtest/gtest.h>

#include "qcor.hpp"
#include "qcor_qsim.hpp"
#include "xacc.hpp"

TEST(QiteWorkflowTester, checkSimple) {
  using namespace qcor;
  auto observable = 0.7071067811865475 * X(0) + 0.7071067811865475 * Z(0);
  xacc::internal_compiler::qpu = xacc::getAccelerator("qpp");
  const int nbSteps = 25;
  const double stepSize = 0.1;
  auto problemModel = QuaSiMo::ModelFactory::createModel(&observable);
  auto workflow = QuaSiMo::getWorkflow(
      "qite", {{"steps", nbSteps}, {"step-size", stepSize}});
  auto result = workflow->execute(problemModel);
  const auto energy = result.get<double>("energy");
  const auto energyAtStep = result.get<std::vector<double>>("exp-vals");
  for (const auto &val : energyAtStep) {
    std::cout << val << "\n";
  }
  // Minimal Energy = -1
  EXPECT_NEAR(energy, -1.0, 1e-2);
  auto finalCircuit =
      result.getPointerLike<xacc::CompositeInstruction>("circuit");
  std::cout << "HOWDY:\n" << finalCircuit->toString() << "\n";
}

// TEST(QiteWorkflowTester, checkDeuteronH2) {
//   using namespace qcor;
//   xacc::set_verbose(true);

//   auto compiler = xacc::getCompiler("xasm");
//   auto ir = compiler->compile(R"(__qpu__ void f(qbit q) { X(q[0]); })", nullptr);
//   auto x = ir->getComposite("f");

//   auto observable = -2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) +
//                     .21829 * Z(0) - 6.125 * Z(1) + 5.907;
//   xacc::internal_compiler::qpu = xacc::getAccelerator("qpp");
//   const int nbSteps = 5;
//   const double stepSize = 0.1;
//   auto problemModel = QuaSiMo::ModelFactory::createModel(x, &observable);
//   auto workflow = QuaSiMo::getWorkflow(
//       "qite", {{"steps", nbSteps}, {"step-size", stepSize}, {"circuit-optimizer", xacc::getIRTransformation("qsearch")}});
//   auto result = workflow->execute(problemModel);
//   const auto energy = result.get<double>("energy");
//   const auto energyAtStep = result.get<std::vector<double>>("exp-vals");
//   for (const auto &val : energyAtStep) {
//     std::cout << val << "\n";
//   }
//   // Minimal Energy = -1
//   // EXPECT_NEAR(energy, -1.0, 1e-2);
//   // auto finalCircuit =
//   // result.getPointerLike<xacc::CompositeInstruction>("circuit"); std::cout <<
//   // "HOWDY:\n" << finalCircuit->toString() << "\n";
// }
int main(int argc, char **argv) {
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}