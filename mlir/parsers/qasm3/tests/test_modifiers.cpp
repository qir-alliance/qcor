#include "gtest/gtest.h"
#include "qcor_mlir_api.hpp"

TEST(qasm3VisitorTester, checkPow) {
  const std::string check_pow = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q;
bit c;

pow(2) @ x q;
measure q -> c;

QCOR_EXPECT_TRUE(c == 0);

reset q;

pow(5) @ x q;
c = measure q;

QCOR_EXPECT_TRUE(c == 1);

gate test r, s {
  x r;
  h s;
}

reset q;
qubit qq;

pow(2) @ test q, qq;

bit xx, yy;
xx = measure q;

QCOR_EXPECT_TRUE(xx == 0);

x qq;
yy = measure qq;
QCOR_EXPECT_TRUE(yy == 1);

qubit a;

s a;
inv @ s a;
bit b;
b = measure a;
QCOR_EXPECT_TRUE(b == 0);

reset a;

bit bb;
pow(2) @ inv @ s a;
measure a -> bb;
QCOR_EXPECT_TRUE(bb == 0);

qubit z, zz;
int count = 0;
for i in [0:100] {
  h z;
  ctrl @ x z, zz;
  bit g[2];
  measure z -> g[0];
  measure zz -> g[1];
  print(g[0], g[1]);
  if (g[0] == 0 && g[1] == 0) {
    count += 1;
  }
  reset z;
  reset zz;
}
print(count);

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