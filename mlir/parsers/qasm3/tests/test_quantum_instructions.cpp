#include "gtest/gtest.h"
#include "qcor_mlir_api.hpp"

TEST(qasm3VisitorTester, checkQuantumBroadcast) {
  const std::string broadcast = R"#(OPENQASM 3;
include "qelib1.inc";
qubit[4] q;
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