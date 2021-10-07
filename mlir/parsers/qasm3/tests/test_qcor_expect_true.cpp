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

TEST(qasm3CompilerTester, checkTestingUtils) {
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
const d = 2;
QCOR_EXPECT_TRUE(d == 2);
const i = 1;
QCOR_EXPECT_TRUE(i == 1);
float[64] f;
f = 1.234;
QCOR_EXPECT_TRUE(f == 1.234);

)#";
  auto mlir =
      qcor::mlir_compile(src, "test", qcor::OutputType::MLIR, true);
  std::cout << "MLIR:\n" << mlir << "\n";

  // We expect false because execution
  // should return 0, anything else is an error code
  EXPECT_FALSE(qcor::execute(src, "test"));

  const std::string src2 = R"#(OPENQASM 3;
include "qelib1.inc";
const d = 2;
QCOR_EXPECT_TRUE(d == 2);
const i = 33;
QCOR_EXPECT_TRUE(i == 1);
)#";
  mlir =
      qcor::mlir_compile(src2, "test", qcor::OutputType::MLIR, true);
  std::cout << "MLIR:\n" << mlir << "\n";

  // We expect true because execution
  // should return 1, 33 not equal to 1
  EXPECT_TRUE(qcor::execute(src2, "test"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}