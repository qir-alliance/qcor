#include "qcor_mlir_api.hpp"
#include "gtest/gtest.h"

namespace {
// returns count of non-overlapping occurrences of 'sub' in 'str'
int countSubstring(const std::string &str, const std::string &sub) {
  if (sub.length() == 0)
    return 0;
  int count = 0;
  for (size_t offset = str.find(sub); offset != std::string::npos;
       offset = str.find(sub, offset + sub.length())) {
    ++count;
  }
  return count;
}
} // namespace

TEST(qasm3VisitorTester, checkCtrlDirectives) {
  const std::string uint_index = R"#(OPENQASM 3;
include "qelib1.inc";

int[64] iterate_value = 0;
int[64] hit_continue_value = 0;
for i in [0:10] {
    iterate_value = i;
    if (i == 5) {
        print("breaking at 5");
        break;
    }
    if (i == 2) {
        hit_continue_value = i;
        print("continuing at 2");
        continue;
    }
    print("i = ", i);
}

QCOR_EXPECT_TRUE(iterate_value == 5);
QCOR_EXPECT_TRUE(hit_continue_value == 2);

print("made it out of the loop");

)#";
  auto mlir = qcor::mlir_compile(uint_index, "uint_index",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  // We're now using Affine and SCF
  EXPECT_EQ(countSubstring(mlir, "affine.for"), 1);
  EXPECT_GT(countSubstring(mlir, "scf.if"), 1);
  EXPECT_FALSE(qcor::execute(uint_index, "uint_index"));
}

TEST(qasm3VisitorTester, checkCtrlDirectivesComplex) {
  const std::string uint_index = R"#(OPENQASM 3;
include "qelib1.inc";

int[64] iterate_value = 0;
int[64] hit_continue_value = 0;
for i in [0:10] {
    iterate_value = i;
    if (i == 5) {
        print("breaking at 5");
        break;
    } else {
      if (i == 3) { 
        print("breaking at 3");
        break;
      }
    }
    if (i == 2) {
      hit_continue_value = i;
      print("continuing at 2");
      continue;
    }

    if (iterate_value == 2) {
      hit_continue_value = 5;
      print("SHOULD NEVER BE HERE!!!");
    } 
    
    print("i = ", i);
}

QCOR_EXPECT_TRUE(hit_continue_value == 2);
// The break at 3 in the else loop will be activated first.
QCOR_EXPECT_TRUE(iterate_value == 3);
print("made it out of the loop");)#";
  auto mlir = qcor::mlir_compile(uint_index, "uint_index",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  // We're now using Affine and SCF
  EXPECT_EQ(countSubstring(mlir, "affine.for"), 1);
  EXPECT_GT(countSubstring(mlir, "scf.if"), 1);
  EXPECT_FALSE(qcor::execute(uint_index, "uint_index"));
}

TEST(qasm3VisitorTester, checkCtrlDirectivesSetBasedForLoop) {
  const std::string uint_index = R"#(OPENQASM 3;
include "qelib1.inc";

int[64] sum_value = 0;
int[64] break_value = 0;
int[64] loop_count = 0;

for val in {1,3,5,7} {
  print("iter: ", val);
  if (val < 4) {
    sum_value += val;
  } else {
    break_value = val;
    break;
  }

  loop_count += 1;
}

print(sum_value);
print(loop_count);
print(break_value);
QCOR_EXPECT_TRUE(sum_value == 4);
QCOR_EXPECT_TRUE(loop_count == 2);
QCOR_EXPECT_TRUE(break_value == 5);)#";
  auto mlir = qcor::mlir_compile(uint_index, "uint_index",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  // Make sure we're using Affine and SCF
  EXPECT_EQ(countSubstring(mlir, "affine.for"), 1);
  EXPECT_GT(countSubstring(mlir, "scf.if"), 1);
  EXPECT_FALSE(qcor::execute(uint_index, "uint_index"));
}

TEST(qasm3VisitorTester, checkCtrlDirectivesWhileLoop) {
  const std::string uint_index = R"#(OPENQASM 3;
include "qelib1.inc";

int[32] i = 0;
int[32] j = 0;
while (i < 10) {
  // Before break
  i += 1;
  if (i == 8) {
    print("Breaking at", i);
    break;
  }
  // After break
  j += 1;
}

print("make to the end, i =", i);
print("make to the end, j =", j);
QCOR_EXPECT_TRUE(i == 8);
QCOR_EXPECT_TRUE(j == 7);
)#";
  auto mlir = qcor::mlir_compile(uint_index, "uint_index",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  // Make sure we're using Affine While loop
  EXPECT_EQ(countSubstring(mlir, "scf.while"), 1);
  EXPECT_FALSE(qcor::execute(uint_index, "uint_index"));
}

TEST(qasm3VisitorTester, checkIqpewithIf) {
  const std::string qasm_code = R"#(OPENQASM 3;
include "qelib1.inc";

// Expected to get 4 bits (iteratively) of 1011 (or 1101 LSB) = 11(decimal):
// phi_est = 11/16 (denom = 16 since we have 4 bits)
// => phi = 2pi * 11/16 = 11pi/8 = 2pi - 5pi/8
// i.e. we estimate the -5*pi/8 angle...
qubit q[2];
const bits_precision = 4;
bit c[bits_precision];

// Prepare the eigen-state: |1>
x q[1];

// First bit
h q[0];
// Controlled rotation: CU^k
for i in [0:8] {
  cphase(-5*pi/8) q[0], q[1];
}
h q[0];
// Measure and reset
measure q[0] -> c[0];
reset q[0];

// Second bit
h q[0];
for i in [0:4] {
  cphase(-5*pi/8) q[0], q[1];
}
// Conditional rotation
if (c[0] == 1) {
  rz(-pi/2) q[0];
}
h q[0];
// Measure and reset
measure q[0] -> c[1];
reset q[0];

// Third bit
h q[0];
for i in [0:2] {
  cphase(-5*pi/8) q[0], q[1];
}
// Conditional rotation
if (c[0] == 1) {
  rz(-pi/4) q[0];
}
if (c[1] == 1) {
  rz(-pi/2) q[0];
}
h q[0];
// Measure and reset
measure q[0] -> c[2];
reset q[0];

// Fourth bit
h q[0];
cphase(-5*pi/8) q[0], q[1];
// Conditional rotation
if (c[0] == 1) {
  rz(-pi/8) q[0];
}
if (c[1] == 1) {
  rz(-pi/4) q[0];
}
if (c[2] == 1) {
  rz(-pi/2) q[0];
}
h q[0];
measure q[0] -> c[3];

print(c[0], c[1], c[2], c[3]);
QCOR_EXPECT_TRUE(c[0] == 1);
QCOR_EXPECT_TRUE(c[1] == 1);
QCOR_EXPECT_TRUE(c[2] == 0);
QCOR_EXPECT_TRUE(c[3] == 1);
)#";
  // Make sure we can compile this in FTQC.
  // i.e., usual if ...
  auto mlir = qcor::mlir_compile(qasm_code, "iqpe",
                                 qcor::OutputType::LLVMIR, false);
  std::cout << mlir << "\n";
  // Execute (FTQC + optimization): validate expected results: 1101
  EXPECT_FALSE(qcor::execute(qasm_code, "iqpe"));
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}