#include "gtest/gtest.h"
#include "qcor_mlir_api.hpp"

TEST(qasm3VisitorTester, checkDeclaration) {
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
print("HELLO WORLD");
int[10] x, y;
int[5] xx=2, yy=1;
qubit q1[6], q2;
bit b1[4]="0100", b2 = "1";
bit k, kk[22];
bool bb = False;
bool m=True, n=bool(xx);
const c = 5.5e3, d=5;
const e = 2.2;
print(c);
print(bb);
print(m);
print(n);
print(xx);
print(b1[1]);
)#";
  auto mlir =
      qcor::mlir_compile("qasm3", src, "test", qcor::OutputType::MLIR, true);
  std::cout << "MLIR:\n" << mlir << "\n";
  qcor::execute("qasm3", src, "test");
}

TEST(qasm3VisitorTester, checkAssignment) {
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
const d = 6;
const layers = 22;
const layers2 = layers / 2;
const t = layers * 3;
const tt = d * 33.3;
const mypi = pi / 2;
const added = layers + t;
const added_diff_types = layers + tt;
int[64] tmp = 10, tmp2 = 33, tmp3 = 22;

tmp += tmp2;
tmp -= tmp3;
tmp *= 2;
tmp /= 2;

int[32] i = 10;
float[32] f;
float[64] ff = 3.14;
bit result;
bit results[2];
creg c[22];
bool b, z;
bool bb = 1;
bool bbb = 0;

print(layers);
print(layers2);
print(t);
print(tt);
print(mypi);
print(added);
print(added_diff_types);
print(tmp);
print(b, z);
print(bb);
print(bbb);
print(f);
print(ff);
)#";
  auto mlir =
      qcor::mlir_compile("qasm3", src, "test", qcor::OutputType::MLIR, true);
  std::cout << "MLIR:\n" << mlir << "\n";
  qcor::execute("qasm3", src, "test");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}

