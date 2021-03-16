#include "qcor_mlir_api.hpp"
#include "gtest/gtest.h"

TEST(qasm3VisitorTester, checkAlias) {
  {
    // Test 1: Alias by indices
    const std::string alias_by_indicies = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q[6];
// myreg[0,1,2] refers to the qubit q[1,3,5]
let myreg = q[1, 3, 5];
// Apply x on qubits in the alias list
for i in [0:3] {
  x myreg[i];
}
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
)#";
    auto mlir = qcor::mlir_compile("qasm3", alias_by_indicies, "test",
                                   qcor::OutputType::MLIR, true);
    std::cout << "MLIR:\n" << mlir << "\n";
    EXPECT_FALSE(qcor::execute("qasm3", alias_by_indicies, "test"));
  }
  // Test 2: Alias by slice:
  {
    const std::string alias_by_slice = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q[6];
// Range without step size
let myreg1 = q[0:3];

// Range with step size (0, 2, 4)
let myreg2 = q[0:2:5];

// Range with negative step:
let myreg3 = q[4:-1:2];

// Range with start = stop
// This is q[5]
let myreg4 = q[5:5];

// Range using negative indexing:
// Last 3 qubits
let myreg5 = q[-4:-1];
)#";
    auto mlir = qcor::mlir_compile("qasm3", alias_by_slice, "test",
                                   qcor::OutputType::MLIR, true);
    std::cout << "MLIR:\n" << mlir << "\n";
    // EXPECT_FALSE(qcor::execute("qasm3", alias_by_slice, "test"));
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
