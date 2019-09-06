#include <gtest/gtest.h>
#include "qcor.hpp"

TEST(bell_state_tester, check_bell_state) {

  xacc::setAccelerator("local-ibm");

  auto bell = [&](qbit q) {
      H(q[0]);
      CX(q[0],q[1]);
      Measure(q[0]);
      Measure(q[1]);
  };

  auto q = qcor::qalloc(2);
  bell(q);

  q->print();

  auto handle = qcor::submit([&](qcor::qpu_handler& qh) {
      qh.execute(bell);
  });

  auto results = qcor::sync(handle);
  results->print();

}

int main(int argc, char **argv) {
  xacc::Initialize();//argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
