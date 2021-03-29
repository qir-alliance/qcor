#include "gtest/gtest.h"
#include "qcor_mlir_api.hpp"

TEST(qasm3VisitorTester, checkPow) {
  const std::string check_pow = R"#(OPENQASM 3;
include "qelib1.inc";

qubit z[2];
int count = 0;
for i in [0:100] {
  h z[0];
  ctrl @ x z[0], z[1];
  bit g[2];
  measure z -> g;

  if (g[0] == 0 && g[1] == 0) {
    count += 1;
  }
  reset z;
}
print(count);
QCOR_EXPECT_TRUE(count > 30);
)#";
  auto mlir = qcor::mlir_compile("qasm3", check_pow, "check_pow",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";

 
  EXPECT_FALSE(qcor::execute("qasm3", check_pow, "check_pow"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}