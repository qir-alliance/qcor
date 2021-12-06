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
#include "qcor_mlir_api.hpp"

TEST(qasm3VisitorTester, checkUintIndexing) {
  const std::string uint_index = R"#(OPENQASM 3;
include "qelib1.inc";

uint[4] b_in = 15;

bool b1 = bool(b_in[0]);
bool b2 = bool(b_in[1]);
bool b3 = bool(b_in[2]);
bool b4 = bool(b_in[3]);

print(b1,b2,b3,b4);
QCOR_EXPECT_TRUE(b1 == 1);
QCOR_EXPECT_TRUE(b2 == 1);
QCOR_EXPECT_TRUE(b3 == 1);
QCOR_EXPECT_TRUE(b4 == 1);

)#";
  auto mlir = qcor::mlir_compile(uint_index, "uint_index",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  EXPECT_FALSE(qcor::execute(uint_index, "uint_index"));
}

TEST(qasm3VisitorTester, checkCastBitToInt) {
  const std::string cast_int = R"#(OPENQASM 3;
include "qelib1.inc";
bit c[4] = "1111";
int[4] t = int[4](c);
// should print 15
print(t);
QCOR_EXPECT_TRUE(t == 15);
)#";
  auto mlir = qcor::mlir_compile(cast_int, "cast_int",
                                 qcor::OutputType::MLIR, false);

  std::cout << "cast_int MLIR:\n" << mlir << "\n";

  EXPECT_FALSE(qcor::execute(cast_int, "cast_int"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}