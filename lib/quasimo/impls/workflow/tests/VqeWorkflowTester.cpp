/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "qcor.hpp"
#include "qcor_qsim.hpp"
#include "xacc.hpp"
#include <gtest/gtest.h>

TEST(VqeWorkflowTest, checkInputComposite) {
  using namespace qcor;
  auto observable = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) +
                    .21829 * Z(0) - 6.125 * Z(1);
  xacc::internal_compiler::qpu = xacc::getAccelerator("qpp");

  auto xasm = xacc::getCompiler("xasm");
  auto tmp = xasm->compile(R"#(__qpu__ void ansatz(qbit q, double theta) {
  X(q[0]);
  exp_i_theta(q, theta, {{"pauli", "X0 Y1 - Y0 X1"}});
  }
)#");
  auto kernel =
      std::make_shared<qcor::CompositeInstruction>(tmp->getComposites()[0]);
  auto H = 5.907 - 2.1433 * X(0) * X(1) - 2.143 * Y(0) * Y(1) + 0.21829 * Z(0) -
           6.125 * Z(1);
  auto problemModel = QuaSiMo::ModelFactory::createModel(kernel, H);
  auto optimizer = createOptimizer("nlopt");
  // Instantiate a VQE workflow with the nlopt optimizer
  auto workflow = QuaSiMo::getWorkflow("vqe", {{"optimizer", optimizer}});
  auto result = workflow->execute(problemModel);
  const auto energy = result.get<double>("energy");
  std::cout << "Min energy: " << energy << "\n";
  EXPECT_NEAR(energy, -1.748, 0.1);
}

int main(int argc, char **argv) {
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}