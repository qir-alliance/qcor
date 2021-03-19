#include "gtest/gtest.h"
#include "qcor_mlir_api.hpp"

TEST(qasm3VisitorTester, checkDeclaration) {
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
int[10] x = 5;
int[10] y;
print(x);
QCOR_EXPECT_TRUE(x == 5);
QCOR_EXPECT_TRUE(y == 0);

int[5] xx=2, yy=1;
QCOR_EXPECT_TRUE(xx == 2);
QCOR_EXPECT_TRUE(yy == 1);

qubit q1[6], q2;
bit b1[4]="0100", b2 = "1";
QCOR_EXPECT_TRUE(b1[0] == 0);
QCOR_EXPECT_TRUE(b1[1] == 1);
QCOR_EXPECT_TRUE(b1[2] == 0);
QCOR_EXPECT_TRUE(b1[3] == 0);

bit k, kk[22];
QCOR_EXPECT_TRUE(k == 0);
QCOR_EXPECT_TRUE(kk[13] == 0);
bool bb = False;
bool m=True, n=bool(xx);
QCOR_EXPECT_TRUE(m == 1);
QCOR_EXPECT_TRUE(bb == 0);
QCOR_EXPECT_TRUE(n == 0);
const c = 5.5e3, d=5;
const e = 2.2;
QCOR_EXPECT_TRUE(c == 5500.0);
QCOR_EXPECT_TRUE(d == 5);
QCOR_EXPECT_TRUE(e == 2.2);
x q2;
k = measure q2;
QCOR_EXPECT_TRUE(k == 1);
for i in [0:22] {
    QCOR_EXPECT_TRUE(kk[i] == 0);
}

float[64] f = 3.14;
float[64] test = 3.14 - f;
QCOR_EXPECT_TRUE(test < .001);
)#";
  auto mlir =
      qcor::mlir_compile("qasm3", src, "test", qcor::OutputType::MLIR, true);
  std::cout << "MLIR:\n" << mlir << "\n";
      auto llvmi =
      qcor::mlir_compile("qasm3", src, "test", qcor::OutputType::LLVMMLIR, true);
  std::cout << "LLVM:\n" << llvmi << "\n";
    auto llvm =
      qcor::mlir_compile("qasm3", src, "test", qcor::OutputType::LLVMIR, true);
  std::cout << "LLVM:\n" << llvm << "\n";
  EXPECT_FALSE(qcor::execute("qasm3", src, "test"));

  const std::string src2 = R"#(OPENQASM 3;
include "qelib1.inc";
bit kk[22];
for i in [0:22] {
    print("should only see 0 bc next statement will fail: ", i);
    QCOR_EXPECT_TRUE(kk[i] == 1);
}
)#";
  auto mlir2 =
      qcor::mlir_compile("qasm3", src2, "test", qcor::OutputType::MLIR, true);
  std::cout << "MLIR:\n" << mlir2 << "\n";
  EXPECT_TRUE(qcor::execute("qasm3", src2, "test"));
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}