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

TEST(qasm3VisitorTester, checkArithmetic) {
  const std::string global_const = R"#(OPENQASM 3;
include "qelib1.inc";
const shots = 1024.0;
float[64] num_parity_ones = 508.0;
float[64] result, test;

result = (shots - num_parity_ones) / shots - num_parity_ones / shots;
test = result - .007812;
QCOR_EXPECT_TRUE(test < .01);
)#";
  auto mlir = qcor::mlir_compile(global_const, "global_const",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  EXPECT_FALSE(qcor::execute(global_const, "global_const"));
}

TEST(qasm3VisitorTester, checkPower) {
  const std::string power_test = R"#(OPENQASM 3;
include "qelib1.inc";
int j = 5;
int y = 2;
int test1 = 2^(j-y);
QCOR_EXPECT_TRUE(test1 == 8);
int test2 = j^(j-y);
QCOR_EXPECT_TRUE(test2 == 125);
)#";
  auto mlir = qcor::mlir_compile(power_test, "power_test",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  EXPECT_FALSE(qcor::execute(power_test, "power_test"));
}

// Check QPE that has complex classical arithmetic:
TEST(qasm3VisitorTester, checkQPE) {
  const std::string qpe_test = R"#(OPENQASM 3;
const n_counting = 3;

// For this example, the oracle is the T gate 
// on the provided qubit
gate oracle b {
    t b;
}

// Inverse QFT subroutine on n_counting qubits
def iqft qubit[n_counting]:qq {
    for i in [0:n_counting/2] {
        swap qq[i], qq[n_counting-i-1];
    }
    for i in [0:n_counting-1] {
        h qq[i];
        int j = i + 1;
        int y = i;
        while (y >= 0) {
            double theta = -pi / (2^(j-y));
            cphase(theta) qq[j], qq[y];
            y -= 1;
        }
    }
    h qq[n_counting-1];
}

// Define some counting qubits
qubit counting[n_counting];

// Allocate the qubit we'll 
// put the initial state on
qubit state;

// We want T |1> = exp(2*i*pi*phase) |1> = exp(i*pi/4)
// compute phase, should be 1 / 8;

// Initialize to |1>
x state;

// Put all others in a uniform superposition
h counting;

// Loop over and create ctrl-U**2k
int repetitions = 1;
for i in [0:n_counting] {
    ctrl @ pow(repetitions) @ oracle counting[i], state;
    repetitions *= 2;
}

// Run inverse QFT 
iqft counting;

// Now lets measure the counting qubits
bit c[n_counting];
measure counting -> c;

// Backend is QPP which is lsb, 
// so return should be 100
print(c);
)#";
  auto mlir = qcor::mlir_compile(qpe_test, "qpe_test",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  EXPECT_FALSE(qcor::execute(qpe_test, "qpe_test"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
