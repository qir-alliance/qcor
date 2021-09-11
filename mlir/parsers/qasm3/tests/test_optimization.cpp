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

TEST(qasm3PassManagerTester, checkIdentityPairRemoval) {
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
qubit[2] q;

x q[0];
x q[0];
cx q[0], q[1];
cx q[0], q[1];
)#";
  auto llvm = qcor::mlir_compile(src, "test", qcor::OutputType::LLVMIR, true);
  std::cout << "LLVM:\n" << llvm << "\n";
  // No instrucions left
  EXPECT_TRUE(llvm.find("__quantum__qis") == std::string::npos);
}

TEST(qasm3PassManagerTester, checkResetSimplification) {
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
qubit[2] q;

reset q[0];
x q[1];
reset q[0];
)#";
  auto llvm =
      qcor::mlir_compile(src, "test_kernel", qcor::OutputType::LLVMIR, false);
  std::cout << "LLVM:\n" << llvm << "\n";

  // Get the main kernel section only
  llvm = llvm.substr(llvm.find("@__internal_mlir_test_kernel"));
  const auto last = llvm.find_first_of("}");
  llvm = llvm.substr(0, last + 1);
  std::cout << "LLVM:\n" << llvm << "\n";
  // One reset and one X:
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis"), 2);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__x"), 1);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__reset"), 1);
}

TEST(qasm3PassManagerTester, checkRotationMerge) {
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
qubit[2] q;

x q[0];
z q[1];
rx(1.2345) q[0];
rz(2.4566) q[1];
)#";
  auto llvm =
      qcor::mlir_compile(src, "test_kernel", qcor::OutputType::LLVMIR, false);
  std::cout << "LLVM:\n" << llvm << "\n";

  // Get the function LLVM only (not __internal_mlir_XXXX, etc.)
  llvm = llvm.substr(llvm.find("@test_kernel"));
  // 2 instrucions left: rx and rz
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis"), 2);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__rx"), 1);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__rz"), 1);
}

TEST(qasm3PassManagerTester, checkSingleQubitGateMergeOpt) {
  // Merge to X == rx(pi)
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
qubit[2] q;
h q[0];
z q[0];
h q[0];
)#";
  auto llvm =
      qcor::mlir_compile(src, "test_kernel", qcor::OutputType::LLVMIR, false);
  std::cout << "LLVM:\n" << llvm << "\n";

  // Get the function LLVM only (not __internal_mlir_XXXX, etc.)
  llvm = llvm.substr(llvm.find("@test_kernel"));
  // 1 instrucions left: rx
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis"), 1);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__rx"), 1);
}

TEST(qasm3PassManagerTester, checkRemoveUnusedQirCalls) {
  // Complete cancellation => remove extract and qalloc as well
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
qubit[2] q;
cx q[0], q[1];
h q[0];
z q[0];
h q[0];
x q[0];
cx q[0], q[1];
)#";
  auto llvm =
      qcor::mlir_compile(src, "test_kernel", qcor::OutputType::LLVMIR, false);
  std::cout << "LLVM:\n" << llvm << "\n";
  // No gates, extract, or alloc/dealloc:
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis"), 0);
  EXPECT_EQ(countSubstring(llvm, "__quantum__rt__array_get_element_ptr_1d"), 0);
  EXPECT_EQ(countSubstring(llvm, "__quantum__rt__qubit_allocate_array"), 0);
  EXPECT_EQ(countSubstring(llvm, "__quantum__rt__qubit_release_array"), 0);
}

TEST(qasm3PassManagerTester, checkInliner) {
  // Inline a call ==> gate cancellation
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
gate oracle b {
  x b;
}

qubit[2] q;
x q[0];
oracle q[0];
)#";
  auto llvm =
      qcor::mlir_compile(src, "test_kernel", qcor::OutputType::LLVMIR, false);
  std::cout << "LLVM:\n" << llvm << "\n";

  // Get the main kernel section only (there is the oracle LLVM section as well)
  llvm = llvm.substr(llvm.find("@test_kernel"));
  const auto last = llvm.find_first_of("}");
  llvm = llvm.substr(0, last + 1);
  std::cout << "LLVM:\n" << llvm << "\n";
  // No gates, extract, or alloc/dealloc:
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis"), 0);
  EXPECT_EQ(countSubstring(llvm, "__quantum__rt__array_get_element_ptr_1d"), 0);
  EXPECT_EQ(countSubstring(llvm, "__quantum__rt__qubit_allocate_array"), 0);
  EXPECT_EQ(countSubstring(llvm, "__quantum__rt__qubit_release_array"), 0);
}

TEST(qasm3PassManagerTester, checkPermuteAndCancel) {
  // Permute rz-cnot ==> gate cancellation
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";

qubit[2] q;

rz(0.123) q[0];
cx q[0], q[1];
rz(-0.123) q[0];
cx q[0], q[1];
)#";
  auto llvm =
      qcor::mlir_compile(src, "test_kernel", qcor::OutputType::LLVMIR, false);
  std::cout << "LLVM:\n" << llvm << "\n";

  // Get the main kernel section only (there is the oracle LLVM section as well)
  llvm = llvm.substr(llvm.find("@test_kernel"));
  const auto last = llvm.find_first_of("}");
  llvm = llvm.substr(0, last + 1);
  std::cout << "LLVM:\n" << llvm << "\n";
  // Cancel all => No gates, extract, or alloc/dealloc:
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis"), 0);
  EXPECT_EQ(countSubstring(llvm, "__quantum__rt__array_get_element_ptr_1d"), 0);
  EXPECT_EQ(countSubstring(llvm, "__quantum__rt__qubit_allocate_array"), 0);
  EXPECT_EQ(countSubstring(llvm, "__quantum__rt__qubit_release_array"), 0);
}

TEST(qasm3PassManagerTester, checkLoopUnroll) {
  // Unroll the loop:
  // cancel all X gates; combine rx
  const std::string src = R"#(OPENQASM 3;
include "stdgates.inc";
qubit[2] q;
for i in [0:10] {
    x q[0];
    rx(0.123) q[1];
}
)#";
  auto llvm =
      qcor::mlir_compile(src, "test_kernel", qcor::OutputType::LLVMIR, false);
  std::cout << "LLVM:\n" << llvm << "\n";

  // Get the main kernel section only
  llvm = llvm.substr(llvm.find("@__internal_mlir_test_kernel"));
  const auto last = llvm.find_first_of("}");
  llvm = llvm.substr(0, last + 1);
  std::cout << "LLVM:\n" << llvm << "\n";
  // Only a single Rx remains (combine all angles)
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis"), 1);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__rx"), 1);
}

TEST(qasm3PassManagerTester, checkLoopUnrollTrotter) {
  // Unroll the loop:
  // Trotter decompose
  const std::string src = R"#(OPENQASM 3;
include "stdgates.inc";
qubit[2] qq;
for i in [0:100] {
    h qq;
    cx qq[0], qq[1];
    rx(0.0123) qq[1];
    cx qq[0], qq[1];
    h qq;
}
)#";
  auto llvm =
      qcor::mlir_compile(src, "test_kernel", qcor::OutputType::LLVMIR, false);
  std::cout << "LLVM:\n" << llvm << "\n";

  // Get the main kernel section only
  llvm = llvm.substr(llvm.find("@__internal_mlir_test_kernel"));
  const auto last = llvm.find_first_of("}");
  llvm = llvm.substr(0, last + 1);
  std::cout << "LLVM:\n" << llvm << "\n";
  // Only a single Rx remains (combine all angles)
  // 2 Hadamard before + 1 CX before
  // 2 Hadamard after + 1 CX after
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis"), 7);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__rx"), 1);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__h"), 4);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__cnot"), 2);
}

TEST(qasm3PassManagerTester, checkLoopUnrollWithInline) {
  // Unroll the loop and inline
  // Trotter decompose
  // Note: using the inv (adjoint) modifier is not supported
  // since it is a runtime feature...
  // hence, we need to make the adjoint explicit.
  const std::string src = R"#(OPENQASM 3;
include "stdgates.inc";
def cnot_ladder() qubit[4]:q {
  h q[0];
  h q[1];
  cx q[0], q[1];
  cx q[1], q[2];
  cx q[2], q[3];
}

def cnot_ladder_inv() qubit[4]:q {
  cx q[2], q[3];
  cx q[1], q[2];
  cx q[0], q[1];
  h q[1];
  h q[0];
}

qubit[4] q;
double theta = 0.01;
for i in [0:100] {
  cnot_ladder q;
  rz(theta) q[3];
  cnot_ladder_inv q;
}
)#";
  auto llvm =
      qcor::mlir_compile(src, "test_kernel", qcor::OutputType::LLVMIR, false);
  std::cout << "LLVM:\n" << llvm << "\n";

  // Get the main kernel section only
  llvm = llvm.substr(llvm.find("@__internal_mlir_test_kernel"));
  const auto last = llvm.find_first_of("}");
  llvm = llvm.substr(0, last + 1);
  std::cout << "LLVM:\n" << llvm << "\n";
  // Only a single Rz remains (combine all angles)
  // 2 Hadamard before + 3 CX before
  // 2 Hadamard after + 3 CX after
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis"), 11);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__rz"), 1);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__h"), 4);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__cnot"), 6);
}

TEST(qasm3PassManagerTester, checkAffineLoopRevert) {
  // Check loop with negative step:
  const std::string src = R"#(OPENQASM 3;
include "stdgates.inc";
def cnot_ladder() qubit[4]:q {
  h q;
  for i in [0:3] {
    cx q[i], q[i + 1];
  }
}

def cnot_ladder_inv() qubit[4]:q {
  for i in [3:-1:0] {
    cx q[i-1], q[i];
  }
  
  h q;
}

qubit[4] q;
double theta = 0.01;
for i in [0:100] {
  cnot_ladder q;
  rz(theta) q[3];
  cnot_ladder_inv q;
}
)#";
  auto llvm =
      qcor::mlir_compile(src, "test_kernel", qcor::OutputType::LLVMIR, false);
  std::cout << "LLVM:\n" << llvm << "\n";

  // Get the main kernel section only
  llvm = llvm.substr(llvm.find("@__internal_mlir_test_kernel"));
  const auto last = llvm.find_first_of("}");
  llvm = llvm.substr(0, last + 1);
  std::cout << "LLVM:\n" << llvm << "\n";
  // Only a single Rz remains (combine all angles)
  // 4 Hadamard before + 3 CX before
  // 4 Hadamard after + 3 CX after
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis"), 15);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__rz"), 1);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__h"), 8);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__cnot"), 6);
}

TEST(qasm3PassManagerTester, checkQubitArrayAlias) {
  {
    // Check SSA value chain with alias
    // h-t-h == rx (t is equiv. to rz)
    const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";

qubit[6] q;
let my_reg = q[1, 3, 5];

h q[1];
t my_reg[0];
h q[1];
)#";
    auto llvm =
        qcor::mlir_compile(src, "test_kernel", qcor::OutputType::LLVMIR, false);
    std::cout << "LLVM:\n" << llvm << "\n";

    // Get the main kernel section only (there is the oracle LLVM section as
    // well)
    llvm = llvm.substr(llvm.find("@__internal_mlir_test_kernel"));
    const auto last = llvm.find_first_of("}");
    llvm = llvm.substr(0, last + 1);
    std::cout << "LLVM:\n" << llvm << "\n";
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis"), 1);
    // One Rx
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__rx"), 1);
  }

  {
    // Check optimization can work with alias array
    const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";

qubit[4] q;
let first_and_last_qubit = q[0] || q[3];

cx q[0], q[3];
cx first_and_last_qubit[0], first_and_last_qubit[1];
)#";
    auto llvm =
        qcor::mlir_compile(src, "test_kernel", qcor::OutputType::LLVMIR, false);
    std::cout << "LLVM:\n" << llvm << "\n";

    // Get the main kernel section only (there is the oracle LLVM section as
    // well)
    llvm = llvm.substr(llvm.find("@__internal_mlir_test_kernel"));
    const auto last = llvm.find_first_of("}");
    llvm = llvm.substr(0, last + 1);
    std::cout << "LLVM:\n" << llvm << "\n";
    // Cancel all => No gates, extract, or alloc/dealloc:
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis"), 0);
    // Make sure all runtime (alias construction) functions are removed as well.
    EXPECT_EQ(countSubstring(llvm, "__quantum__"), 0);
  }
}

TEST(qasm3PassManagerTester, checkConstEvalLoopUnroll) {
  {
    // Unroll the loop with const vars as loop bounds
    const std::string src = R"#(OPENQASM 3;
include "stdgates.inc";

const n = 3;
qubit qb;

for i in [0:2*n] {
 x qb;
}
)#";
    auto llvm =
        qcor::mlir_compile(src, "test_kernel", qcor::OutputType::LLVMIR, false);
    std::cout << "LLVM:\n" << llvm << "\n";

    // Get the main kernel section only
    llvm = llvm.substr(llvm.find("@__internal_mlir_test_kernel"));
    const auto last = llvm.find_first_of("}");
    llvm = llvm.substr(0, last + 1);
    std::cout << "LLVM:\n" << llvm << "\n";
    // Cancel all
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis"), 0);
  }

  {
    // Unroll the loop with const vars as loop bounds
    const std::string src = R"#(OPENQASM 3;
include "stdgates.inc";

const n = 3;
qubit qb;

for i in [0:2*n + 1] {
 x qb;
}
)#";
    auto llvm = qcor::mlir_compile(src, "test_kernel1",
                                   qcor::OutputType::LLVMIR, false);
    std::cout << "LLVM:\n" << llvm << "\n";

    // Get the main kernel section only
    llvm = llvm.substr(llvm.find("@__internal_mlir_test_kernel1"));
    const auto last = llvm.find_first_of("}");
    llvm = llvm.substr(0, last + 1);
    std::cout << "LLVM:\n" << llvm << "\n";
    // One X gate left
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__x"), 1);
  }
}

TEST(qasm3PassManagerTester, checkConditionalBlock) {
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
bit c;
qubit[2] q;

h q[0];
x q[1];
c = measure q[1];
if (c == 1) {
  // Always execute:
  z q[0];
}

h q[0];

// This should be: H - Z - H == X 
// Check that the two H's before and after 'if'
// don't connect.
// i.e., checking that we don't accidentally cancel the two H gates
// => left with Z => measure 0 vs. expected 1.
bit d;
d = measure q[0];
print("measure =", d);
QCOR_EXPECT_TRUE(d == 1);
)#";
  auto llvm =
      qcor::mlir_compile(src, "test_kernel1", qcor::OutputType::LLVMIR, false);
  std::cout << "LLVM:\n" << llvm << "\n";

  // Get the main kernel section only
  llvm = llvm.substr(llvm.find("@__internal_mlir_test_kernel1"));
  const auto last = llvm.find_first_of("}");
  llvm = llvm.substr(0, last + 1);
  std::cout << "LLVM:\n" << llvm << "\n";
  // 2 Hadamard gates, 1 Z gate
  // (Z gate in the conditional block)
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__h"), 2);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__z"), 1);
  EXPECT_FALSE(qcor::execute(src, "test_kernel1"));
}

TEST(qasm3PassManagerTester, checkCPhaseMerge) {
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
  auto llvm =
      qcor::mlir_compile(qasm_code, "test_kernel1", qcor::OutputType::LLVMIR, false);
  std::cout << "LLVM:\n" << llvm << "\n";

  // Get the main kernel section only
  llvm = llvm.substr(llvm.find("@__internal_mlir_test_kernel1"));
  const auto last = llvm.find_first_of("}");
  llvm = llvm.substr(0, last + 1);
  std::cout << "LLVM:\n" << llvm << "\n";
  // All CPhase gates in for loops have been unrolled and then merged.
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__cphase"), 4);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}