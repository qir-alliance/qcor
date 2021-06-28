#include "gtest/gtest.h"
#include "qcor_mlir_api.hpp"

TEST(qasm3VisitorTester, checkSubroutine1) {
  const std::string sub1 = R"#(OPENQASM 3;
include "qelib1.inc";
const n = 10;
def parity(bit[n]:cin) -> bit {
  bit c;
  for i in [0: n-1] {
    c ^= cin[i];
  }
  return c;
}

bit b[n] = "1110001010";
bit p;
p = parity(b);
QCOR_EXPECT_TRUE(p == 1);
print(p);


bit b1[n] = "1111001010";
bit p1;
p1 = parity(b1);
QCOR_EXPECT_TRUE(p1 == 0);
print(p1);
)#";
  auto mlir = qcor::mlir_compile(sub1, "check_parity_sub",
                                 qcor::OutputType::MLIR, false);

  std::cout << mlir << "\n";
  EXPECT_FALSE(qcor::execute(sub1, "check_parity_sub"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}