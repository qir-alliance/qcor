/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#if __GNUC__ >= 5
// Disable GCC 5's -Wsuggest-override and -Wsign-compare warnings in gtest
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wsuggest-override"
# pragma GCC diagnostic ignored "-Wsign-compare"
#endif
#include "gtest/gtest.h"
#include "qir-qrt.hpp"

TEST(QirQrtRefCountTester, checkArray) {
  auto test_array = __quantum__rt__array_create_1d(8, 2);
  EXPECT_EQ(test_array->size(), 2);
  EXPECT_EQ(test_array->element_size(), 8);
  EXPECT_EQ(test_array->ref_count(), 1);
  // Tracker should detect the array is leaking (no dealloc)
  EXPECT_TRUE(qcor::internal::AllocationTracker::get().checkLeak());
  __quantum__rt__array_update_reference_count(test_array, 3);
  EXPECT_EQ(test_array->ref_count(), 4);
  for (int i = 0; i < 4; ++i) {
    __quantum__rt__array_update_reference_count(test_array, -1);
  }

  // Should be dealloc'ed by now...
  EXPECT_FALSE(qcor::internal::AllocationTracker::get().checkLeak());
}

TEST(QirQrtRefCountTester, checkTuple) {
  // No leak
  EXPECT_FALSE(qcor::internal::AllocationTracker::get().checkLeak());
  auto tuple = __quantum__rt__tuple_create(sizeof(double) + sizeof(void*));
  auto tuple_header = TupleHeader::getHeader(tuple);
  EXPECT_EQ(tuple_header->ref_count(), 1);
  // Leak
  EXPECT_TRUE(qcor::internal::AllocationTracker::get().checkLeak());
  __quantum__rt__tuple_update_reference_count(tuple, -1);
  // Gone by now
  EXPECT_FALSE(qcor::internal::AllocationTracker::get().checkLeak());
}

#if __GNUC__ >= 5
# pragma GCC diagnostic pop
#endif

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}