#include <gtest/gtest.h>

#include "LibIntWrapper.hpp"

TEST(LibIntWrapperTester, checkSimple) {

  std::string geom = R"geom(2

H          0.00000        0.00000        0.00000
H          0.00000        0.00000        0.7474
)geom";

  libintwrapper::LibIntWrapper libint;
  libint.generate({{"basis", "sto-3g"}, {"geometry", geom}});

  auto aok = libint.getAOKinetic();
  double expectedKData[] = {0.760031876642567, 0.232569901630748,
                            0.232569901630748, 0.760031876642567};
  Eigen::TensorMap<Eigen::Tensor<double, 2>> expectedK(expectedKData, 2, 2);
  Eigen::Tensor<double, 0> sum1 = expectedK.sum();
  Eigen::Tensor<double, 0> sum2 = aok.sum();
  EXPECT_NEAR(sum1(0), sum2(0), 1e-12);

  auto aov = libint.getAOPotential();
  double expectedVData[] = {-1.87620978630359, -1.1832748909716,
                            -1.1832748909716, -1.87620978630359};
  Eigen::TensorMap<Eigen::Tensor<double, 2>> expectedV(expectedVData, 2, 2);
  sum1 = expectedV.sum();
  sum2 = aov.sum();
  EXPECT_NEAR(sum1(0), sum2(0), 1e-12);

  auto eri = libint.getERI();
  double expectedERIData[] = {
      0.774605929806267, 0.440379924864223, 0.440379924864223,
      0.567214683616369, 0.440379924864223, 0.292709697347553,
      0.292709697347553, 0.440379924864223, 0.440379924864223,
      0.292709697347553, 0.292709697347553, 0.440379924864223,
      0.567214683616369, 0.440379924864223, 0.440379924864223,
      0.774605929806267};

  Eigen::TensorMap<Eigen::Tensor<double, 4>> expectedERI(expectedERIData, 2, 2,
                                                         2, 2);
  sum1 = expectedERI.sum();
  sum2 = eri.sum();
  EXPECT_NEAR(sum1(0), sum2(0), 1e-12);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
