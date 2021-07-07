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
  
  // Get the main kernel section only (there is the oracle LLVM section as well)
  llvm = llvm.substr(llvm.find("@test_kernel"));
  const auto last = llvm.find_first_of("}");
  llvm = llvm.substr(0, last + 1);
  std::cout << "LLVM:\n" << llvm << "\n";
  // Only a single Rx remains (combine all angles)
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis"), 1);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__rx"), 1);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}