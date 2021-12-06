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
#include "gtest/gtest.h"
#include "objective_function.hpp"
#include "qcor_optimizer.hpp"

using namespace qcor;

TEST(OperatorTester, checkSimple) {
  ObjectiveFunction obj(
      [](std::vector<double> x, std::vector<double>& dx) {
        if (!dx.empty()) {
          dx[0] = 2 * x[0];
        }
        return x[0] * x[0] - 5.0;
      },
      1);
  Optimizer optimizer("nlopt");
  auto [opt_val, opt_params] = optimizer.optimize(obj);
  EXPECT_NEAR(opt_val, -5.0, 1e-6);

  Optimizer optimizer2("nlopt", {{"optimizer", "l-bfgs"}});
  auto [opt_val2, opt_params2] = optimizer2.optimize(obj);
  EXPECT_NEAR(opt_val, -5.0, 1e-6);
}

#include "xacc.hpp"

int main(int argc, char** argv) {
  xacc::Initialize(argc, argv);
  xacc::set_verbose(true);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
