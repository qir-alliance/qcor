#include <gtest/gtest.h>

#include "nlopt_optimizer.hpp"

using namespace qcor;

TEST(NLOptimizerTester, checkSimple) {

  NLOptimizer optimizer;

  OptFunction f([](const std::vector<double>& x) {return x[0]*x[0]+5;},1);

  EXPECT_EQ(1,f.dimensions());

  auto result = optimizer.optimize(f);

  EXPECT_EQ(result.first, 5.0);
  EXPECT_EQ(result.second, std::vector<double>{0.0});

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
