#include <gtest/gtest.h>
#include "XACC.hpp"
#include "YPulse.hpp"

using namespace xacc;

using namespace xacc::quantum;

TEST(YPulseTester, emptyTest) {

    // NOW Test it somehow...
  YPulse y({1});
  EXPECT_EQ("YPulse", y.name());
  EXPECT_EQ(1, y.bits()[0]);

  y.setOption("pulse_id", "hello");
  EXPECT_EQ("hello", y.getOption("pulse_id").toString());
//   EXPECT_EQ("null", y.getOptions()["pulse_id"].toString());
//   EXPECT_EQ("d1", y.getOptions()["ch"].toString());

}

int main(int argc, char** argv) {
    xacc::Initialize();
    ::testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();
    xacc::Finalize();
    return ret;
}
