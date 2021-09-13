#include "clifford_gate_utils.hpp"
#include <cassert>
#include <cmath>
namespace {
double mod_2pi(double theta) {
  while (theta > M_PI or theta <= -M_PI) {
    if (theta > M_PI) {
      theta = theta - 2 * M_PI;
    } else if (theta <= -M_PI) {
      theta = theta + 2 * M_PI;
    }
  }
  assert(theta >= -M_PI && theta <= M_PI);
  return theta;
}
} // namespace
namespace qcor {
namespace utils {
GenRot_t computeRotationInPauliFrame(const GenRot_t &in_rot,
                                     PauliLabel in_newPauli,
                                     PauliLabel &io_netPauli) {
  auto [theta1, theta2, theta3] = in_rot;

  if (io_netPauli == PauliLabel::X || io_netPauli == PauliLabel::Z) {
    theta2 *= -1.0;
  }
  if (io_netPauli == PauliLabel::X || io_netPauli == PauliLabel::Y) {
    theta3 *= -1.0;
    theta1 *= -1.0;
  }

  // if x or y
  if (in_newPauli == PauliLabel::X || io_netPauli == PauliLabel::Y) {
    theta1 = -theta1 + M_PI;
    theta2 = theta2 + M_PI;
  }
  // if y or z
  if (in_newPauli == PauliLabel::Y || io_netPauli == PauliLabel::Z) {
    theta1 = theta1 + M_PI;
  }

  // make everything between - pi and pi
  theta1 = mod_2pi(theta1);
  theta2 = mod_2pi(theta2);
  theta3 = mod_2pi(theta3);

  return std::make_tuple(theta1, theta2, theta3);
}
} // namespace utils
} // namespace qcor