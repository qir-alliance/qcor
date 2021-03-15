#include "gtest/gtest.h"
#include "qcor_mlir_api.hpp"

TEST(qasm3VisitorTester, checkAlias) {
  std::cout << "HOWDY\n";
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q[6];
// myreg[0] refers to the qubit q[1]
let myreg = q[1, 3, 5];
)#";
  auto mlir =
      qcor::mlir_compile("qasm3", src, "test", qcor::OutputType::MLIR, true);
  std::cout << "MLIR:\n" << mlir << "\n";
  qcor::execute("qasm3", src, "test");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
