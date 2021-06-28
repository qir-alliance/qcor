#include "gtest/gtest.h"
#include "qcor_mlir_api.hpp"

TEST(qasm3CompilerTester, checkAssignment) {
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
const d = 2;
const dd = d^5;
const fff = 2.0;
const ffff = fff^5;
int[32] i = 2;

QCOR_EXPECT_TRUE(i == 2);
QCOR_EXPECT_TRUE(d == 2);
QCOR_EXPECT_TRUE(dd == 32);
QCOR_EXPECT_TRUE(fff == 2.0);
QCOR_EXPECT_TRUE(ffff == 32.0);

const layers = 22;
const layers2 = layers / 2;
const t = layers * 3;
const tt = d * 33.3;
const mypi = pi / 2;
const added = layers + t;
const added_diff_types = layers + tt;
int[64] tmp = 10, tmp2 = 33, tmp3 = 22;

QCOR_EXPECT_TRUE(layers == 22);
QCOR_EXPECT_TRUE(layers2 == 11);
QCOR_EXPECT_TRUE(t == 66);
QCOR_EXPECT_TRUE(tt == 66.6);

// FIXME May need to add QCOR_COMPARE_FLOATS
//print(mypi);
// QCOR_EXPECT_TRUE(mypi == 1.570796);

QCOR_EXPECT_TRUE(added == 88);
QCOR_EXPECT_TRUE(added_diff_types == 88.6);
QCOR_EXPECT_TRUE(tmp == 10);
QCOR_EXPECT_TRUE(tmp2 == 33);
QCOR_EXPECT_TRUE(tmp3 == 22);

tmp += tmp2;
tmp -= tmp3;
tmp *= 2;
tmp /= 2;

QCOR_EXPECT_TRUE(tmp == 21);

int[32] ii = 10;
float[32] f;
float[64] ff = 3.14;
bit result;
bit results[2];
creg c[22];
bool b, z;
bool bb = 1;
bool bbb = 0;

QCOR_EXPECT_TRUE(ii == 10);
QCOR_EXPECT_TRUE(result == 0);
QCOR_EXPECT_TRUE(results[0] == 0);
QCOR_EXPECT_TRUE(results[1] == 0);
QCOR_EXPECT_TRUE(b == 0);
QCOR_EXPECT_TRUE(z == 0);
QCOR_EXPECT_TRUE(bb == 1);
QCOR_EXPECT_TRUE(bbb == 0);
QCOR_EXPECT_TRUE(f == 0.0);
QCOR_EXPECT_TRUE(ff == 3.14);

)#";
  auto mlir =
      qcor::mlir_compile(src, "test", qcor::OutputType::MLIR, true);
  std::cout << "MLIR:\n" << mlir << "\n";
  EXPECT_FALSE(qcor::execute(src, "test"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}