/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
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
qubit qq[2];
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

TEST(qasm3VisitorTester, checkSubroutineAdder) {
  const std::string sub1 = R"#(OPENQASM 3;
gate ccx a,b,c
{
  h c;
  cx b,c; tdg c;
  cx a,c; t c;
  cx b,c; tdg c;
  cx a,c; t b; t c; h c;
  cx a,b; t a; tdg b;
  cx a,b;
}

gate majority a, b, c {
  cx c, b;
  cx c, a;
  ccx a, b, c;
}

gate unmaj a, b, c {
  ccx a, b, c;
  cx c, a;
  cx a, b;
}

qubit cin;
qubit a[8];
qubit b[8];
qubit cout;
bit ans[9];
// Input values:
uint[8] a_in = 5;  
uint[8] b_in = 4; 

for i in [0:8] {  
  if (bool(a_in[i])) {
    x a[i];
  }
  if (bool(b_in[i])) {
    x b[i];
  }
}
// add a to b, storing result in b
majority cin, b[0], a[0];

for i in [0: 7] { 
  majority a[i], b[i + 1], a[i + 1]; 
}

cx a[7], cout;

for i in [6: -1: -1] { 
  unmaj a[i], b[i+1], a[i+1]; 
}
unmaj cin, b[0], a[0];

measure b[0:7] -> ans[0:7];
measure cout[0] -> ans[8];)#";
  auto llvm_ir = qcor::mlir_compile(sub1, "check_deuteron",
                                    qcor::OutputType::LLVMIR, false);

  std::cout << llvm_ir << "\n";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}