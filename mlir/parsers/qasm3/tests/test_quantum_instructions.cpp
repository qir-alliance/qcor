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
#include "gtest/gtest.h"
#include "qcor_mlir_api.hpp"

TEST(qasm3VisitorTester, checkQuantumBroadcast) {
  const std::string broadcast = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q[4];
x q;
bit m[4];
m = measure q;

for i in [0:4] {
    print(m[i]);
    QCOR_EXPECT_TRUE(m[i] == 1);
}
)#";
  auto mlir = qcor::mlir_compile(broadcast, "broadcast",
                                 qcor::OutputType::MLIR, false);

  EXPECT_FALSE(qcor::execute(broadcast, "for_stmt"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}