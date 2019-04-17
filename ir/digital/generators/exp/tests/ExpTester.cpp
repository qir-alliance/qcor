#include "exp.hpp"
#include "XACC.hpp"
#include <gtest/gtest.h>

#include "qcor.hpp"

using namespace xacc;
using namespace qcor::instructions;

TEST(ExpTester, checkSimple) {

  // NOW Test it somehow...
  Exp exp;
  auto f =
      exp.generate({{"pauli","Y0 X1 X2 X3"}});

      std::cout << "F:\n" << f->toString() << "\n";

}

int main(int argc, char **argv) {
  qcor::Initialize(argc, argv);
//   xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
//   xacc::Finalize();
  return ret;
}
