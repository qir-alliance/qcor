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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
