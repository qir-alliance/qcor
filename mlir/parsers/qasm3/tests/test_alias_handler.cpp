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
)#";
  auto mlir = qcor::mlir_compile("qasm3", alias_by_indicies, "test",
                                 qcor::OutputType::MLIR, true);
  std::cout << "MLIR:\n" << mlir << "\n";
  EXPECT_FALSE(qcor::execute("qasm3", alias_by_indicies, "test"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
