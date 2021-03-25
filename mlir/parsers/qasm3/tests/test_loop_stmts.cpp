#include "gtest/gtest.h"
#include "qcor_mlir_api.hpp"

TEST(qasm3VisitorTester, checkLoops) {
  const std::string for_stmt = R"#(OPENQASM 3;
include "qelib1.inc";

int[32] loop_count = 0;
for i in {11,22,33} {
    print(i);
    loop_count += 1;
}
QCOR_EXPECT_TRUE(loop_count == 3);

loop_count = 0;
for i in [0:10] {
    print(i);
    loop_count += 1;
    
}
QCOR_EXPECT_TRUE(loop_count == 10);
loop_count = 0;

for j in [0:2:4] {
    print("steps:", j);
    loop_count += 1;
}

QCOR_EXPECT_TRUE(loop_count == 2);
loop_count = 0;

for j in [0:4] {
    print("j in 0:4", j);
    loop_count += 1;
}

QCOR_EXPECT_TRUE(loop_count == 4);
loop_count = 0;

for i in [0:4] {
 for j in {1,2,3} {
     print(i,j);
     loop_count += 1;
 }
}
QCOR_EXPECT_TRUE(loop_count == 12);


)#";
  auto mlir = qcor::mlir_compile("qasm3", for_stmt, "for_stmt",
                                 qcor::OutputType::MLIR, false);
  std::cout << "for_stmt MLIR:\n" << mlir << "\n";
  EXPECT_FALSE(qcor::execute("qasm3", for_stmt, "for_stmt"));

  const std::string while_stmt = R"#(OPENQASM 3;
include "qelib1.inc";
int[32] i = 0;
while (i < 10) {
  print(i);
  i += 1;
}
QCOR_EXPECT_TRUE(i == 10);
)#";
  auto mlir2 = qcor::mlir_compile("qasm3", while_stmt, "while_stmt",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir2 << "\n";
  EXPECT_FALSE(qcor::execute("qasm3", while_stmt, "while_stmt"));

    const std::string decrement = R"#(OPENQASM 3;
include "qelib1.inc";
for j in [10:-1:0] {
  print(j);
}
)#";
  auto mlir3 = qcor::mlir_compile("qasm3", decrement, "decrement",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir3 << "\n";
  EXPECT_FALSE(qcor::execute("qasm3", decrement, "decrement"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}