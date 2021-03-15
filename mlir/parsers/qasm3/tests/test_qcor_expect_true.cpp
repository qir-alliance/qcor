#include "gtest/gtest.h"
#include "qcor_mlir_api.hpp"

TEST(qasm3CompilerTester, checkTestingUtils) {
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
const d = 2;
QCOR_EXPECT_TRUE(d == 2);
const i = 1;
QCOR_EXPECT_TRUE(i == 1);
)#";
  auto mlir =
      qcor::mlir_compile("qasm3", src, "test", qcor::OutputType::MLIR, true);
  std::cout << "MLIR:\n" << mlir << "\n";

  // We expect false because execution
  // should return 0, anything else is an error code
  EXPECT_FALSE(qcor::execute("qasm3", src, "test"));

  const std::string src2 = R"#(OPENQASM 3;
include "qelib1.inc";
const d = 2;
QCOR_EXPECT_TRUE(d == 2);
const i = 33;
QCOR_EXPECT_TRUE(i == 1);
)#";
  mlir =
      qcor::mlir_compile("qasm3", src2, "test", qcor::OutputType::MLIR, true);
  std::cout << "MLIR:\n" << mlir << "\n";

  // We expect true because execution
  // should return 1, 33 not equal to 1
  EXPECT_TRUE(qcor::execute("qasm3", src2, "test"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}