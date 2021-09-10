#include "gtest/gtest.h"
#include "qcor_mlir_api.hpp"

TEST(qasm3VisitorTester, checkDeclaration) {
    const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
print("HELLO WORLD");
int[10] x, y;
int[5] xx=2, yy=1;
qubit[6] q1;
qubit q2;
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
            qcor::mlir_compile(src, "test", qcor::OutputType::MLIR, true);
    std::cout << "MLIR:\n" << mlir << "\n";
    qcor::execute(src, "test");
}

TEST(qasm3VisitorTester, checkAssignment) {
    const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
const d = 6;
const dd = d^2;
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
            qcor::mlir_compile(src, "test", qcor::OutputType::MLIR, true);
    std::cout << "MLIR:\n" << mlir << "\n";
    qcor::execute(src, "test");
}

TEST(qasm3VisitorTester, checkMeasurements) {
    const std::string measure_test = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q;
qubit[2] qq;

bit r, s;
r = measure q;
s = measure qq[0];

bit rr[2];
rr[0] = measure qq[0];

bit xx[4];
qubit[4] qqq;
xx = measure qqq;


)#";
    auto mlir = qcor::mlir_compile(measure_test, "measure_test",
                                   qcor::OutputType::MLIR, false);
    std::cout << "MLIR:\n" << mlir << "\n";
    // qcor::execute(measure_test, "test");
}

TEST(qasm3VisitorTester, checkQuantumInsts) {
    const std::string qinst_test = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q;
h q;
ry(2.2) q;

qubit[2] qq;
x qq[0];
CX qq[0], qq[1];
U(0.1,0.2,0.3) qq[1];
cx q, qq[1];

)#";
    auto mlir = qcor::mlir_compile(qinst_test, "qinst_test",
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
    auto mlir = qcor::mlir_compile(for_stmt, "for_stmt",
                                   qcor::OutputType::MLIR, false);
    std::cout << "for_stmt MLIR:\n" << mlir << "\n";
    qcor::execute(for_stmt, "for_stmt");
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
    auto mlir = qcor::mlir_compile(uint_index, "uint_index",
                                   qcor::OutputType::MLIR, false);
    std::cout << mlir << "\n";
    qcor::execute(uint_index, "uint_index");
}

TEST(qasm3VisitorTester, checkIfStmt) {
    const std::string if_stmt = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q;
qubit s;
const layers = 2;
bit cc[2];
qubit[2] qq;

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
    auto mlir = qcor::mlir_compile(if_stmt, "if_stmt",
                                   qcor::OutputType::MLIR, false);
    std::cout << mlir << "\n";
}

TEST(qasm3VisitorTester, checkSecondIfStmt) {
    const std::string if_stmt = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q;
qubit s;
qubit[2] qqq;
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
    auto mlir = qcor::mlir_compile(if_stmt, "if_stmt",
                                   qcor::OutputType::MLIR, false);
    std::cout << mlir << "\n";
    qcor::execute(if_stmt, "if_stmt");
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
    auto mlir = qcor::mlir_compile(complex_if, "complex_if",
                                   qcor::OutputType::MLIR, false);
    std::cout << mlir << "\n";
    qcor::execute(complex_if, "complex_if");
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
    auto mlir = qcor::mlir_compile(while_stmt, "while_stmt",
                                   qcor::OutputType::MLIR, false);
    std::cout << mlir << "\n";
    qcor::execute(while_stmt, "while_stmt");
}

TEST(qasm3VisitorTester, checkSubroutine) {
    const std::string subroutine_test = R"#(OPENQASM 3;
include "qelib1.inc";
def xmeasure qubit:q -> bit { h q; return measure q; }
qubit q;
qubit[2] qq;
bit r, rr[2];

rr[0] = xmeasure q;
r = xmeasure qq[0];
)#";
    auto mlir = qcor::mlir_compile(subroutine_test, "subroutine_test",
                                   qcor::OutputType::MLIR, false);

    std::cout << "subroutine_test MLIR:\n" << mlir << "\n";
}

TEST(qasm3VisitorTester, checkSubroutine2) {
    const std::string subroutine_test = R"#(OPENQASM 3;
include "qelib1.inc";
def xcheck qubit[4]:d, qubit:a -> bit {
  // reset a;
  for i in [1: 3] cx d[i], a;
  return measure a;
}
qubit q;
const n = 10;
def parity(bit[n]:cin) -> bit {
  bit c;
  for i in [0: n-1] {
    c ^= cin[i];
  }
  return c;
}
)#";
    auto mlir = qcor::mlir_compile(subroutine_test, "subroutine_test",
                                   qcor::OutputType::MLIR, false);

    std::cout << "subroutine_test MLIR:\n" << mlir << "\n";
}

TEST(qasm3VisitorTester, checkSubroutine3) {
    const std::string subroutine_test = R"#(OPENQASM 3;
include "qelib1.inc";
const n = 10;
def xmeasure qubit:q -> bit { h q; return measure q; }
def ymeasure qubit:q -> bit { s q; h q; return measure q; }

def pauli_measurement(bit[2*n]:spec) qubit[n]:q -> bit {
  bit b;
  for i in [0: n - 1] {
    bit temp;
    if(spec[i]==1 && spec[n+i]==0) { temp = xmeasure q[i]; }
    if(spec[i]==0 && spec[n+i]==1) { temp = measure q[i]; }
    if(spec[i]==1 && spec[n+i]==1) { temp = ymeasure q[i]; }
    b ^= temp;
  }
  return b;
}
)#";
    auto mlir = qcor::mlir_compile(subroutine_test, "subroutine_test",
                                   qcor::OutputType::MLIR, false);

    std::cout << "subroutine_test MLIR:\n" << mlir << "\n";
}

TEST(qasm3VisitorTester, checkSubroutine4) {
    const std::string subroutine_test = R"#(OPENQASM 3;
include "qelib1.inc";
const buffer_size = 30;

def ymeasure qubit:q -> bit { s q; h q; return measure q; }

def test(int[32]:addr) qubit:q, qubit[buffer_size]:buffer {
  bit outcome;
  cy buffer[addr], q;
  outcome = ymeasure buffer[addr];
  if(outcome == 1) ry(pi / 2) q;
}
)#";
    auto mlir = qcor::mlir_compile(subroutine_test, "subroutine_test",
                                   qcor::OutputType::MLIR, false);

    std::cout << "subroutine_test MLIR:\n" << mlir << "\n";
}

// TO IMPLEMENT

TEST(qasm3VisitorTester, checkMeasureRange) {
    const std::string meas_range = R"#(OPENQASM 3;
include "qelib1.inc";
qubit[4] a;
qubit[4] b;
bit ans[5];
measure b[1:3] -> ans[1:3];
ans[1:3] = measure b[1:3];
)#";
    auto mlir = qcor::mlir_compile(meas_range, "meas_range",
                                   qcor::OutputType::MLIR, false);

    std::cout << "meas_range MLIR:\n" << mlir << "\n";
}

TEST(qasm3VisitorTester, checkGate) {
    const std::string gate_def = R"#(OPENQASM 3;
include "qelib1.inc";
gate cphase(x) a, b
{
  U(0, 0, x / 2) a;
  CX a, b;
  U(0, 0, -x / 2) b;
  CX a, b;
  U(0, 0, x / 2) b;
}
qubit[2] q;
qubit[3] r;
cphase(pi / 2) q[0], q[1];

// qubit s, t;
// cphase(pi / 2) s, t;

)#";
    auto mlir = qcor::mlir_compile(gate_def, "gate_def",
                                   qcor::OutputType::MLIR, false);

    std::cout << "gate_def MLIR:\n" << mlir << "\n";

    qcor::execute(gate_def, "gate_def");

    std::cout << "LLVM:\n"
              << qcor::mlir_compile(gate_def, "gate_def",
                                    qcor::OutputType::LLVMIR, false)
              << "\n";
}

TEST(qasm3VisitorTester, checkCastBitToInt) {
    const std::string cast_int = R"#(OPENQASM 3;
include "qelib1.inc";
bit c[4] = "1111";
int[4] t = int[4](c);
// should print 15
print(t);
)#";
    auto mlir = qcor::mlir_compile(cast_int, "cast_int",
                                   qcor::OutputType::MLIR, false);

    std::cout << "cast_int MLIR:\n" << mlir << "\n";

    auto mlir2 = qcor::mlir_compile(cast_int, "cast_int",
                                    qcor::OutputType::LLVMIR, false);

    std::cout << "cast_int MLIR:\n" << mlir2 << "\n";
    qcor::execute(cast_int, "cast_int");
}

TEST(qasm3VisitorTester, checkQuantumBroadcast) {
    const std::string broadcast = R"#(OPENQASM 3;
include "qelib1.inc";
qubit[4] q
qubit[4] r;
qubit a;
qubit b;
qubit[2] aa;
qubit[2] bb;
h q;

cx q, r;
cx a, aa;
cx bb, b;

reset r;

)#";
    auto mlir = qcor::mlir_compile(broadcast, "broadcast",
                                   qcor::OutputType::MLIR, false);

    std::cout << "cast_int MLIR:\n" << mlir << "\n";
}

TEST(qasm3VisitorTester, checkBuiltInMath) {
    const std::string built_in_math = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q;
float[64] x = pi - arccos(3/5);
print(x);
rz(x) q;
)#";
    auto mlir = qcor::mlir_compile(built_in_math, "built_in_math",
                                   qcor::OutputType::MLIR, false);

    std::cout << "cast_int MLIR:\n" << mlir << "\n";
    auto llvm = qcor::mlir_compile(built_in_math, "built_in_math",
                                   qcor::OutputType::LLVMIR, false);
    std::cout << "LLVM:\n" << llvm << "\n";
}

TEST(qasm3VisitorTester, checkWhile2) {
    const std::string while2 = R"#(OPENQASM 3;
include "qelib1.inc";
bit flags[2] = "11";
print(int[2](flags));
// while(int[2](flags) != 0) {
//   print(flags[0]);

// }
)#";
    auto mlir = qcor::mlir_compile(while2, "while2",
                                   qcor::OutputType::MLIR, false);

    std::cout << "cast_int MLIR:\n" << mlir << "\n";
    auto llvm = qcor::mlir_compile(while2, "while2",
                                   qcor::OutputType::LLVMIR, false);
    std::cout << "LLVM:\n" << llvm << "\n";
    qcor::execute(while2, "while2");

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();
    return ret;
}
