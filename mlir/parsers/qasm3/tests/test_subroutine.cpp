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

TEST(qasm3VisitorTester, checkSubroutineDeuteron) {
  const std::string sub1 = R"#(OPENQASM 3;
include "stdgates.inc";

const shots = 1024;
// State-preparation:
def ansatz(float[64]:theta) qubit[2]:q {
    x q[0];
    ry(theta) q[1];
    cx q[1], q[0];
}

def deuteron(float[64]:theta) qubit[2]:q -> float[64] {
    bit first, second;
    float[64] num_parity_ones = 0.0;
    float[64] result;
    for i in [0:shots] {
        ansatz(theta) q;
        // Change measurement basis
        h q;
        // Measure
        first = measure q[0];
        second = measure q[1];
        if (first != second) {
            num_parity_ones += 1.0;
        }
        // Reset
        reset q;
    }

    // Compute expectation value
    result = (shots - num_parity_ones) / shots - num_parity_ones / shots;
    return result;
}

float[64] theta, exp_val;
qubit[2] qq;
// Try a theta value:
const step = 2 * pi / 19;
theta = -pi + step;
print("Theta = ", theta);
exp_val = deuteron(theta) qq;
print("Avg <X0X1> = ", exp_val);
)#";
  auto mlir = qcor::mlir_compile(sub1, "check_deuteron",
                                 qcor::OutputType::MLIR, false);

  std::cout << mlir << "\n";
  EXPECT_FALSE(qcor::execute(sub1, "check_deuteron"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}