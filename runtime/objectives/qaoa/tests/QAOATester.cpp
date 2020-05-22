#include "qcor.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_quantum_gate_api.hpp"
#include "xacc_service.hpp"
#include <gtest/gtest.h>

using namespace xacc;

TEST(QAOATester, checkSimple) {
  // TODO
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
