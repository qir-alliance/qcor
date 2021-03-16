#include "gtest/gtest.h"
#include "qcor_mlir_api.hpp"

TEST(qasm3VisitorTester, checkAlias) {
  std::cout << "HOWDY\n";
  const std::string src = R"#(OPENQASM 3;
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
  auto mlir =
      qcor::mlir_compile("qasm3", src, "test", qcor::OutputType::MLIR, true);
  std::cout << "MLIR:\n" << mlir << "\n";
  EXPECT_FALSE(qcor::execute("qasm3", src, "test"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
