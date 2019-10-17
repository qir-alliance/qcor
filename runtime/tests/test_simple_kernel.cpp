#include <gtest/gtest.h>
#include "qcor.hpp"



TEST(test_task_initiate, checkVariations) {

  auto bell = [&](qbit q) {
      H(q[0]);
      CX(q[0],q[1]);
      Measure(q[0]);
      Measure(q[1]);
  };

  // Test just executing the kernel
  auto q = qcor::qalloc(2);
  bell(q);
  q->print();
  EXPECT_TRUE(q->getMeasurementCounts().count("00"));
  EXPECT_TRUE(q->getMeasurementCounts().count("11"));

  // Now test with submit()
  auto handle = qcor::submit([&](qcor::qpu_handler& qh) {
      qh.execute(bell);
  });
  auto results = qcor::sync(handle);
  results->print();
  EXPECT_TRUE(results->getMeasurementCounts().count("00"));
  EXPECT_TRUE(results->getMeasurementCounts().count("11"));

  // Now test with task initiate
//   auto handle1 = qcor::taskInitiate(bell);
//   auto results2 = qcor::sync(handle1);
//   results2->print();
//   EXPECT_TRUE(results->getMeasurementCounts().count("00"));
//   EXPECT_TRUE(results->getMeasurementCounts().count("11"));

}

int main(int argc, char **argv) {
  xacc::Initialize();//argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
