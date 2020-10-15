#include <gtest/gtest.h>

TEST(TimeSeriesQpeTester, checkSimple) {
  // TODO:
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
