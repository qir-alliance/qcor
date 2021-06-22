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
      qcor::mlir_compile("qasm3", src, "test", qcor::OutputType::LLVMIR, true);
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
      qcor::mlir_compile("qasm3", src, "test_kernel", qcor::OutputType::LLVMIR, false);
  std::cout << "LLVM:\n" << llvm << "\n";
  
  // Get the function LLVM only (not __internal_mlir_XXXX, etc.)
  llvm = llvm.substr(llvm.find("@test_kernel"));
  // 2 instrucions left: rx and rz
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis"), 2);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__rx"), 1);
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis__rz"), 1);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}