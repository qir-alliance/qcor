#include "xacc.hpp"
#include "xacc_service.hpp"
#include <gtest/gtest.h>
#include <fstream>


TEST(MirrorCircuitTester, checkSimple) {
  // TODO
}


int main(int argc, char **argv) {
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
