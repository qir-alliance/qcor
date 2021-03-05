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

TEST(qasm3VisitorTester, checkMeasurements) {
  const std::string measure_test = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q;
qubit qq[2];

bit r, s;
r = measure q;
s = measure qq[0];

bit rr[2];
rr[0] = measure qq[0];

bit xx[4];
qubit qqq[4];
xx = measure qqq;

// bit y, yy[2];
// measure q -> y;
// measure qq -> yy;
// measure qq[0] -> y;
)#";
  auto mlir = qcor::mlir_compile("qasm3", measure_test, "measure_test",
                                 qcor::OutputType::MLIR, false);
  std::cout << "MLIR:\n" << mlir << "\n";
  // qcor::execute("qasm3", measure_test, "test");
}

TEST(qasm3VisitorTester, checkQuantumInsts) {
  const std::string qinst_test = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q;
h q;
ry(2.2) q;

qubit qq[2];
x qq[0];
CX qq[0], qq[1];
U(0.1,0.2,0.3) qq[1];
cx q, qq[1];

)#";
  auto mlir = qcor::mlir_compile("qasm3", qinst_test, "qinst_test",
                                 qcor::OutputType::MLIR, false);
  std::cout << "MLIR:\n" << mlir << "\n";
}

TEST(qasm3VisitorTester, checkLoopStmt) {
  const std::string for_stmt = R"#(OPENQASM 3;
include "qelib1.inc";

for i in {11,22,33} {
    print(i);
}

for i in [0:10] {
    print(i);
    
}
for j in [0:2:4] {
    print("steps:", j);
}


for j in [0:4] {
    print("j in 0:4", j);
}

for i in [0:4] {
 for j in {1,2,3} {
     print(i,j);
 }
}
)#";
  auto mlir = qcor::mlir_compile("qasm3", for_stmt, "for_stmt",
                                 qcor::OutputType::MLIR, false);
  std::cout << "for_stmt MLIR:\n" << mlir << "\n";
  qcor::execute("qasm3", for_stmt, "for_stmt");
}

TEST(qasm3VisitorTester, checkUintIndexing) {
  const std::string uint_index = R"#(OPENQASM 3;
include "qelib1.inc";

uint[4] b_in = 15;

bool b1 = bool(b_in[0]);
bool b2 = bool(b_in[1]);
bool b3 = bool(b_in[2]);
bool b4 = bool(b_in[3]);

print(b1,b2,b3,b4);
)#";
  auto mlir = qcor::mlir_compile("qasm3", uint_index, "uint_index",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  qcor::execute("qasm3", uint_index, "uint_index");
}

TEST(qasm3VisitorTester, checkIfStmt) {
  const std::string if_stmt = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q, s;//, qq[2];
const layers = 2;
bit cc[2];
qubit qq[2];

bit c;
c = measure q;
cc[0] = measure qq[0];

if (c == 1) {
    z s;
} else {
  print("c was a 0");
}

if (layers == 2) {
    print("should be here, layers is 2");
    z s;
} 


cc[1] = measure qq[1];
if ( cc[1] == 1) {
  ry(2.2) s;
}


)#";
  auto mlir = qcor::mlir_compile("qasm3", if_stmt, "if_stmt",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
}

TEST(qasm3VisitorTester, checkSecondIfStmt) {
  const std::string if_stmt = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q, s, qqq[2];
bit c;

if (!c) {
 print("you should see me");
}
x q;
c = measure q;
if (c == 1) {
  print("hi");
  ry(2.2) s;
} 

c = measure qqq[0];
print("hi world");

)#";
  auto mlir = qcor::mlir_compile("qasm3", if_stmt, "if_stmt",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  qcor::execute("qasm3", if_stmt, "if_stmt");
}

TEST(qasm3VisitorTester, checkIfStmt3) {
  const std::string complex_if = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q;
const n = 10;
int[32] i = 3;

bit temp;
if(temp==0 && i==3) {
  print("we are here"); 
  temp = measure q; 
}

)#";
  auto mlir = qcor::mlir_compile("qasm3", complex_if, "complex_if",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  qcor::execute("qasm3", complex_if, "complex_if");

}

TEST(qasm3VisitorTester, checkWhile) {
  const std::string while_stmt = R"#(OPENQASM 3;
include "qelib1.inc";
int[32] i = 0;
while (i < 10) {
  print(i);
  i += 1;
}
)#";
  auto mlir = qcor::mlir_compile("qasm3", while_stmt, "while_stmt",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  qcor::execute("qasm3", while_stmt, "while_stmt");
}
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
