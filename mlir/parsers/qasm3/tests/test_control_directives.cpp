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

// Check Affine-SCF constructs
TEST(qasm3VisitorTester, checkCFG_AffineScf) {
  const std::string qasm_code = R"#(OPENQASM 3;
include "qelib1.inc";

int[64] iterate_value = 0;
int[64] value_5 = 0;
int[64] value_2 = 0;
for i in [0:10] {
    iterate_value = i;
    if (i == 5) {
        print("Iterate over 5");
        value_5 = 5;
    }
    if (i == 2) {
        print("Iterate over 2");
        value_2 = 2;
       
    }
    print("i = ", i);
}

QCOR_EXPECT_TRUE(iterate_value == 9);
QCOR_EXPECT_TRUE(value_5 == 5);
QCOR_EXPECT_TRUE(value_2 == 2);
print("made it out of the loop");
)#";
  auto mlir = qcor::mlir_compile(qasm_code, "affine_scf",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  // 1 for loop, 2 if blocks
  EXPECT_EQ(countSubstring(mlir, "affine.for"), 1);
  EXPECT_EQ(countSubstring(mlir, "scf.if"), 2);
  EXPECT_FALSE(qcor::execute(qasm_code, "affine_scf"));
}

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
  EXPECT_FALSE(qcor::execute(uint_index, "uint_index"));
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}