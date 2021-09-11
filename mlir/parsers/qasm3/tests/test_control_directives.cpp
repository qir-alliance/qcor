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

TEST(qasm3VisitorTester, checkEarlyReturn) {
  const std::string uint_index = R"#(OPENQASM 3;
include "qelib1.inc";

def generate_number(int[64]: count) -> int[64] {
  for i in [0:count] {
    if (i > 10) {
      print("Return at ", i);
      return 5;
    }
    print("i =", i);
  }

  print("make it to the end");
  return 1;  
}

int[64] val1 = generate_number(4);
print("Result 1 =", val1);
QCOR_EXPECT_TRUE(val1 == 1);
// Call it with 20 -> activate the early return
int[64] val2 = generate_number(20);
print("Result 2 =", val2);
QCOR_EXPECT_TRUE(val2 == 5);
)#";
  auto mlir = qcor::mlir_compile(uint_index, "uint_index",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  EXPECT_FALSE(qcor::execute(uint_index, "uint_index"));
}

// Coverage for https://github.com/ORNL-QCI/qcor/issues/211 as well
TEST(qasm3VisitorTester, checkEarlyReturnWith211) {
  const std::string uint_index = R"#(OPENQASM 3;
include "qelib1.inc";

def generate_number(int[64]: count) -> int[64] {
  for i in [0:count] {
    if (i > 10) {
      print("Return at ", i);
      return 3;
    }
    print("i =", i);
  }

  print("make it to the end");
  return count;  
}

int[64] arg_val = 7;
int[64] val1 = generate_number(arg_val);
print("Result 1 =", val1);
QCOR_EXPECT_TRUE(val1 == arg_val);
// Call it with 20 -> activate the early return
int[64] val2 = generate_number(20);
print("Result 2 =", val2);
QCOR_EXPECT_TRUE(val2 == 3);
)#";
  auto mlir = qcor::mlir_compile(uint_index, "uint_index",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  EXPECT_FALSE(qcor::execute(uint_index, "uint_index"));
}

TEST(qasm3VisitorTester, checkEarlyReturnNestedLoop) {
  const std::string uint_index = R"#(OPENQASM 3;
include "qelib1.inc";

def generate_number(int[64]: break_value, int[64]: max_run) -> int[64] {
  int[64] run_count = 0;
  for i in [0:10] {
    for j in [0:10] {
      run_count += 1;
      if (run_count > max_run) {
        print("Exceeding max run count of", max_run);
        return 3;
      }
      
      if (i == j && i > break_value) {
        print("Return at i = ", i);
        print("Return at j = ", j);
        return run_count;
      }

      print("i =", i);
      print("j =", j);
    }
    print("Out of inner loop");
  }

  print("make it to the end");
  return 0;  
}

// Case 1: run to the end.
int[64] val = generate_number(10, 100);
print("Result =", val);
QCOR_EXPECT_TRUE(val == 0);

// Case 2: Return @ (i == j && i > break_value) 
// i = 0: 10; i = 1: 10; i = 2: j = 0, 1, 2 
// => 23 runs (return run_count in this path)
val = generate_number(1, 100);
print("Result =", val);
QCOR_EXPECT_TRUE(val == 23);

// Case 3: return due to max_run limit
// limit to 20 (less than 23) => return value 3
val = generate_number(1, 20);
print("Result =", val);
QCOR_EXPECT_TRUE(val == 3);
)#";
  auto mlir = qcor::mlir_compile(uint_index, "uint_index",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  EXPECT_FALSE(qcor::execute(uint_index, "uint_index"));
}

// Nesting for and while loops
TEST(qasm3VisitorTester, checkEarlyReturnNestedWhileLoop) {
  const std::string uint_index = R"#(OPENQASM 3;
include "qelib1.inc";

def generate_number(int[64]: break_value, int[64]: max_run) -> int[64] {
  int[64] run_count = 0;
  int[64] i = 0;
  while(run_count < max_run) {
    for j in [0:10] {
      run_count += 1;
      if (i == j && i > break_value) {
        print("Return at i = ", i);
        print("Return at j = ", j);
        return run_count;
      }

      print("i =", i);
      print("j =", j);
    }
    i += 1;
    print("Out of inner loop");
  }

  print("make it to the end");
  return 0;  
}

// Case 1: early return @ (i == j && i > break_value) 
// => 23 runs (return run_count in this path)
int[64] val = generate_number(1, 100);
print("Result =", val);
QCOR_EXPECT_TRUE(val == 23);

// Case 2: return at the end
// Make it to the end since run_count will hit 20 
// before hitting the early return condition.
val = generate_number(1, 20);
print("Result =", val);
QCOR_EXPECT_TRUE(val == 0);
)#";
  auto mlir = qcor::mlir_compile(uint_index, "uint_index",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  EXPECT_FALSE(qcor::execute(uint_index, "uint_index"));
}

// Test a complex construction with nesting of different loop types,
// break, return, etc...
TEST(qasm3VisitorTester, checkControlDirectiveKitchenSink) {
  const std::string uint_index = R"#(OPENQASM 3;
include "qelib1.inc";

def find_number(int[64]: target, int[64]: max_run) -> int[64] {
  int[64] run_count = 0;
  for i in { 1, 3, 5, 7, 9} {
    while(run_count < max_run) {
      for j in [0:10] {
        run_count += 1;
        if (i == target && j == i) {
          print("Return at i = ", i);
          print("Return at j = ", j);
          print("Return run_count = ", run_count);
          return run_count;
        }

        print("i =", i);
        print("j =", j);
      }
      // Finish the searching [0->10], break the while loop
      // This is a weird construction,
      // just for testing.
      break;
    }
    print("Make it out of the loop");
    print("run_count = ", run_count);
  }
  
  print("Not found");
  return 0;
}

int[64] val = find_number(3, 100);
print("Result  =", val);
// Find number 3 at 14 iterations.
QCOR_EXPECT_TRUE(val == 14);


val = find_number(2, 100);
print("Result =", val);
// Cannot find number 2 in the set:
QCOR_EXPECT_TRUE(val == 0);

val = find_number(7, 20);
print("Result =", val);
// Cannot find number 7 with only 20 iterations
QCOR_EXPECT_TRUE(val == 0);

val = find_number(7, 100);
print("Result =", val);
// But will find it at iteration 38...
QCOR_EXPECT_TRUE(val == 38);
)#";
  auto mlir = qcor::mlir_compile(uint_index, "uint_index",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  EXPECT_FALSE(qcor::execute(uint_index, "uint_index"));
}

TEST(qasm3VisitorTester, checkIqpewithIf) {
  const std::string qasm_code = R"#(OPENQASM 3;
include "qelib1.inc";

// Expected to get 4 bits (iteratively) of 1011 (or 1101 LSB) = 11(decimal):
// phi_est = 11/16 (denom = 16 since we have 4 bits)
// => phi = 2pi * 11/16 = 11pi/8 = 2pi - 5pi/8
// i.e. we estimate the -5*pi/8 angle...
qubit[2] q;
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
  auto mlir =
      qcor::mlir_compile(qasm_code, "iqpe", qcor::OutputType::LLVMIR, false);
  std::cout << mlir << "\n";
  // Execute (FTQC + optimization): validate expected results: 1101
  EXPECT_FALSE(qcor::execute(qasm_code, "iqpe"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}