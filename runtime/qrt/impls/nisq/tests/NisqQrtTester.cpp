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
#include "qcor.hpp"
#include <gtest/gtest.h>
#include "xacc_service.hpp"
#include "Circuit.hpp"

TEST(NisqQrtTester, checkExpInst) {
  ::quantum::initialize("qpp", "empty");
  const std::string obs_str =
      "5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1";
  auto observable = qcor::createOperator(obs_str);
  auto qreg = qalloc(2);
  qreg.setName("q");
  ::quantum::exp(qreg, 1.0, observable);
  std::cout << "HOWDY\n"
            << ::quantum::qrt_impl->get_current_program()->toString() << "\n";
  // Get the XACC reference implementation
  auto exp = std::dynamic_pointer_cast<xacc::quantum::Circuit>(
      xacc::getService<xacc::Instruction>("exp_i_theta"));
  EXPECT_TRUE(exp->expand({{"pauli", obs_str}}));
  auto evaled = exp->operator()({0.5});
  EXPECT_EQ(evaled->nInstructions(),
            ::quantum::qrt_impl->get_current_program()->nInstructions());
  for (int i = 0; i < evaled->nInstructions(); ++i) {
    auto ref_inst = evaled->getInstruction(i);
    auto qrt_inst =
        ::quantum::qrt_impl->get_current_program()->getInstruction(i);
    std::cout << ref_inst->toString() << "\n";
    std::cout << qrt_inst->toString() << "\n";
    EXPECT_EQ(ref_inst->name(), qrt_inst->name());
    EXPECT_EQ(ref_inst->bits(), qrt_inst->bits());
    if (!ref_inst->getParameters().empty()) {
      EXPECT_EQ(ref_inst->getParameters().size(),
                qrt_inst->getParameters().size());
      for (int j = 0; j < ref_inst->getParameters().size(); ++j) {
        EXPECT_NEAR(ref_inst->getParameters()[j].as<double>(),
                    qrt_inst->getParameters()[j].as<double>(), 1e-9);
      }
    }
  }
}

TEST(NisqQrtTester, checkResetInstSim) {
  ::quantum::initialize("qpp", "empty");
  ::quantum::set_shots(1024);
  auto qreg = qalloc(1);
  qreg.setName("q");
  ::quantum::h({"q", 0});
  ::quantum::reset({"q", 0});
  ::quantum::x({"q", 0});
  ::quantum::mz({"q", 0});
  std::cout << "HOWDY\n"
            << ::quantum::qrt_impl->get_current_program()->toString() << "\n";
  ::quantum::submit(qreg.results());
  qreg.print();
  // Because of reset after H, qubit -> 0 then becomes 1 after X.
  EXPECT_EQ(qreg.counts()["1"], 1024);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
