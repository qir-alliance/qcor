#include "qcor.hpp"
#include <gtest/gtest.h>
#include "xacc_service.hpp"

TEST(FtqcQrtTester, checkSimple) {
  ::quantum::set_qrt("ftqc");
  ::quantum::initialize("qpp", "empty");
  auto qreg = qalloc(1);
  qreg.setName("q");
  ::quantum::set_current_buffer(qreg.results());
  const auto nTests = 100;
  int nZeros = 0;
  int nOnes = 0;
  for (int i = 0; i < nTests; ++i) {
    ::quantum::h({"q", 0});
    if (::quantum::mz({"q", 0})) {
      ++nOnes;
    } else {
      ++nZeros;
    }
  }
  std::cout << "Number of one: " << nOnes << "\n";
  std::cout << "Number of zero: " << nZeros << "\n";
  EXPECT_GT(nZeros, 0);
  EXPECT_GT(nOnes, 0);
}

TEST(FtqcQrtTester, checkReset) {
  ::quantum::set_qrt("ftqc");
  ::quantum::initialize("qpp", "empty");
  auto qreg = qalloc(1);
  qreg.setName("q");
  ::quantum::set_current_buffer(qreg.results());
  const auto nTests = 100;
  int nZeros = 0;
  int nOnes = 0;
  for (int i = 0; i < nTests; ++i) {
    ::quantum::h({"q", 0});
    // Apply reset
    ::quantum::reset({"q", 0});
    ::quantum::x({"q", 0});
    if (::quantum::mz({"q", 0})) {
      ++nOnes;
    } else {
      ++nZeros;
    }
  }
  std::cout << "Number of one: " << nOnes << "\n";
  std::cout << "Number of zero: " << nZeros << "\n";
  EXPECT_EQ(nZeros, 0);
  // Expect all 1's due to reset.
  EXPECT_EQ(nOnes, 100);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
