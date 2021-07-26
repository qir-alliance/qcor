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
qubit q[2];

x q[0];
x q[0];
cx q[0], q[1];
cx q[0], q[1];
)#";
  auto llvm =
      qcor::mlir_compile(src, "test", qcor::OutputType::LLVMIR, true);
  std::cout << "LLVM:\n" << llvm << "\n";
  // No instrucions left
  EXPECT_TRUE(llvm.find("__quantum__qis") == std::string::npos);
}

TEST(qasm3PassManagerTester, checkResetSimplification) {
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q[2];

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
qubit q[2];

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
qubit q[2];
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
qubit q[2];
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

qubit q[2];
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

qubit q[2];

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
qubit q[2];
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
qubit qq[2];
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

qubit q[4];
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

qubit q[4];
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

qubit q[6];
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

qubit q[4];
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
    auto llvm =
        qcor::mlir_compile(src, "test_kernel1", qcor::OutputType::LLVMIR, false);
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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}