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
#include "qcor_mlir_api.hpp"
#include "gtest/gtest.h"

TEST(qasm3VisitorTester, checkAlias) {
  const std::string alias_by_indicies = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q[6];
// Test 1: Alias by indices
// myreg[0,1,2] refers to the qubit q[1,3,5]
let myreg = q[1, 3, 5];
// Apply x on qubits in the alias list
// Use broadcast to make sure it work
x myreg;
// Measure all qubits
bit m[6];
m = measure q;

for i in [0:6] {
  print(m[i]);
}
QCOR_EXPECT_TRUE(m[0] == 0);
QCOR_EXPECT_TRUE(m[1] == 1);
QCOR_EXPECT_TRUE(m[2] == 0);
QCOR_EXPECT_TRUE(m[3] == 1);
QCOR_EXPECT_TRUE(m[4] == 0);
QCOR_EXPECT_TRUE(m[5] == 1);
// Reset q to start next test
reset q;
bit m1[6];
m1 = measure q;
QCOR_EXPECT_TRUE(m1[0] == 0);
QCOR_EXPECT_TRUE(m1[1] == 0);
QCOR_EXPECT_TRUE(m1[2] == 0);
QCOR_EXPECT_TRUE(m1[3] == 0);
QCOR_EXPECT_TRUE(m1[4] == 0);
QCOR_EXPECT_TRUE(m1[5] == 0);

// Test 2: Alias by slice:
// 0, 1, 2, 3 (inclusive)
let myreg1 = q[0:3];
x myreg1;
// Measure all qubits
bit m2[6];
m2 = measure q;

for j in [0:6] {
  print(m2[j]);
}
QCOR_EXPECT_TRUE(m2[0] == 1);
QCOR_EXPECT_TRUE(m2[1] == 1);
QCOR_EXPECT_TRUE(m2[2] == 1);
QCOR_EXPECT_TRUE(m2[3] == 1);
QCOR_EXPECT_TRUE(m2[4] == 0);
QCOR_EXPECT_TRUE(m2[5] == 0);

// Reset q to start next test
reset q;

// Range with step size (0, 2, 4)
let myreg2 = q[0:2:5];
x myreg2;
// Measure all qubits
bit m3[6];
m3 = measure q;

for k in [0:6] {
  print(m3[k]);
}
QCOR_EXPECT_TRUE(m3[0] == 1);
QCOR_EXPECT_TRUE(m3[1] == 0);
QCOR_EXPECT_TRUE(m3[2] == 1);
QCOR_EXPECT_TRUE(m3[3] == 0);
QCOR_EXPECT_TRUE(m3[4] == 1);
QCOR_EXPECT_TRUE(m3[5] == 0);

// Reset q to start next test
reset q;
// Range with negative step:
// 4, 3, 2
let myreg3 = q[4:-1:2];
x myreg3;
// Measure all qubits
bit m4[6];
m4 = measure q;

for i1 in [0:6] {
  print(m4[i1]);
}
QCOR_EXPECT_TRUE(m4[0] == 0);
QCOR_EXPECT_TRUE(m4[1] == 0);
QCOR_EXPECT_TRUE(m4[2] == 1);
QCOR_EXPECT_TRUE(m4[3] == 1);
QCOR_EXPECT_TRUE(m4[4] == 1);
QCOR_EXPECT_TRUE(m4[5] == 0);

// Reset q to start next test
reset q;
// Range with start = stop
// This is q[5]
let myreg4 = q[5:5];
x myreg4;
// Measure all qubits
bit m5[6];
m5 = measure q;

for i2 in [0:6] {
  print(m5[i2]);
}
QCOR_EXPECT_TRUE(m5[0] == 0);
QCOR_EXPECT_TRUE(m5[1] == 0);
QCOR_EXPECT_TRUE(m5[2] == 0);
QCOR_EXPECT_TRUE(m5[3] == 0);
QCOR_EXPECT_TRUE(m5[4] == 0);
QCOR_EXPECT_TRUE(m5[5] == 1);

// Reset q to start next test
reset q;
// Range using negative indexing:
// Last 3 qubits
let myreg5 = q[-3:-1];
x myreg5;
// Measure all qubits
bit m6[6];
m6 = measure q;

for i3 in [0:6] {
  print(m6[i3]);
}
QCOR_EXPECT_TRUE(m6[0] == 0);
QCOR_EXPECT_TRUE(m6[1] == 0);
QCOR_EXPECT_TRUE(m6[2] == 0);
QCOR_EXPECT_TRUE(m6[3] == 1);
QCOR_EXPECT_TRUE(m6[4] == 1);
QCOR_EXPECT_TRUE(m6[5] == 1);

// Test concatenate:
// Reset q to start next test
reset q;
let even_set = q[0:2:5];
let odd_set = q[1:2:5];
let both = even_set || odd_set;
x both;
// Measure all qubits
bit m7[6];
m7 = measure q;

for i in [0:6] {
  print(m7[i]);
}
// All ones
QCOR_EXPECT_TRUE(m7[0] == 1);
QCOR_EXPECT_TRUE(m7[1] == 1);
QCOR_EXPECT_TRUE(m7[2] == 1);
QCOR_EXPECT_TRUE(m7[3] == 1);
QCOR_EXPECT_TRUE(m7[4] == 1);
QCOR_EXPECT_TRUE(m7[5] == 1);

// Test concatenate complex
// Reset q to start next test
reset q;
// Inline concat: 0, 3, 1, 5
let concat_inline = q[0:3:5] || q[1:4:5];
x concat_inline;
// Measure all qubits
bit m8[6];
m8 = measure q;

for i in [0:6] {
  print(m8[i]);
}
// 0, 3, 1, 5 ==> 1
QCOR_EXPECT_TRUE(m8[0] == 1);
QCOR_EXPECT_TRUE(m8[1] == 1);
QCOR_EXPECT_TRUE(m8[2] == 0);
QCOR_EXPECT_TRUE(m8[3] == 1);
QCOR_EXPECT_TRUE(m8[4] == 0);
QCOR_EXPECT_TRUE(m8[5] == 1);

// Reset q to start next test
reset q;
// Multi-concat: 0, 1, 2, 4, 5 (no 3)
let concat_multiple = q[0:1:2] || q[4:4] || q[5];
x concat_multiple;
// Measure all qubits
bit m9[6];
m9 = measure q;

for i in [0:6] {
  print(m9[i]);
}
// All ones except 3
QCOR_EXPECT_TRUE(m9[0] == 1);
QCOR_EXPECT_TRUE(m9[1] == 1);
QCOR_EXPECT_TRUE(m9[2] == 1);
QCOR_EXPECT_TRUE(m9[3] == 0);
QCOR_EXPECT_TRUE(m9[4] == 1);
QCOR_EXPECT_TRUE(m9[5] == 1);

// Check one-qubit alias:
// Reset q to start next test
reset q;
let first_qubit = q[0];
let last_qubit = q[5];
x first_qubit;
ctrl @ x first_qubit, last_qubit;
// Measure all qubits
bit m10[6];
m10 = measure q;

// First and last qubits are 1:
QCOR_EXPECT_TRUE(m10[0] == 1);
QCOR_EXPECT_TRUE(m10[1] == 0);
QCOR_EXPECT_TRUE(m10[2] == 0);
QCOR_EXPECT_TRUE(m10[3] == 0);
QCOR_EXPECT_TRUE(m10[4] == 0);
QCOR_EXPECT_TRUE(m10[5] == 1);
)#";
  auto mlir = qcor::mlir_compile(alias_by_indicies, "test",
                                 qcor::OutputType::MLIR, true);
  std::cout << "MLIR:\n" << mlir << "\n";
  EXPECT_FALSE(qcor::execute(alias_by_indicies, "test"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
