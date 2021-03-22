#include "gtest/gtest.h"
#include "qcor_mlir_api.hpp"

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
  auto mlir = qcor::mlir_compile("qasm3", uint_index, "uint_index",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  EXPECT_FALSE(qcor::execute("qasm3", uint_index, "uint_index"));
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}