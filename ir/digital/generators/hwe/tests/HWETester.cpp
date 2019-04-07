#include "hwe.hpp"
#include "XACC.hpp"
#include <gtest/gtest.h>

#include "qcor.hpp"

using namespace xacc;
using namespace qcor::instructions;

TEST(HWETester, checkSimple) {

  // NOW Test it somehow...
  HWE hwe;
  auto f =
      hwe.generate({
          {"n-qubits", 2},
          {"layers", 1},
          {"coupling", std::vector<std::pair<int,int>>{{0, 1}}}
          });

//   std::cout << f->toString() << "\n";
  EXPECT_EQ(11, f->nInstructions());
  EXPECT_EQ(10, f->nParameters());

  auto m = std::map<std::string,InstructionParameter>{
          {"n-qubits", 4},
          {"layers", 1},
          {"coupling", std::vector<std::pair<int,int>>{{0, 1},{1,2},{2,3}}}
          };
  auto f2 = hwe.generate(m);
    // std::cout << f->toString() << "\n";
  EXPECT_EQ(23, f2->nInstructions());
  EXPECT_EQ(20, f2->nParameters());
}

int main(int argc, char **argv) {
  qcor::Initialize(argc, argv);
//   xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
//   xacc::Finalize();
  return ret;
}
