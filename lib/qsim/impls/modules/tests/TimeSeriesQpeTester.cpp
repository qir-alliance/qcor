#include "utils/prony_method.hpp"
#include <gtest/gtest.h>
#include "xacc.hpp"

TEST(TimeSeriesQpeTester, checkPronyMethod) {
  auto x_vec = xacc::linspace(0.0, 1.0, 11);
  std::vector<std::complex<double>> y_vec;
  constexpr std::complex<double> I{0.0, 1.0};
  for (const auto &xVal : x_vec) {
    y_vec.emplace_back(
        0.5 * std::exp(I * xVal * 3.0) + 0.3 * std::exp(I * xVal * 5.0) +
        0.15 * std::exp(I * xVal * 1.5) + 0.1 * std::exp(I * xVal * 4.0) +
        0.05 * std::exp(I * xVal * 1.2));
  }
  auto result = qcor::utils::pronyFit(y_vec);
  size_t idx = 0;
  const std::vector<double> expectedAmpls { 0.05, 0.1, 0.15, 0.3, 0.5 };
  const std::vector<double> expectedFreqs { 0.12, 0.4, 0.15, 0.5, 0.3 };

  for (const auto& [ampl, phase] : result) {
    const auto freq =  std::arg(phase);
    const auto amplitude = std::abs(ampl);
    EXPECT_NEAR(freq, expectedFreqs[idx], 1e-3);
    EXPECT_NEAR(amplitude, expectedAmpls[idx], 1e-3);
    std::cout << "A = " << amplitude << "; " << "Freq = " << freq << "\n";
    idx++;
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
