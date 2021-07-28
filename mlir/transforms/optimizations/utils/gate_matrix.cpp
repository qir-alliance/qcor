#include "gate_matrix.hpp"
// Turn off Eigen warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wunused-function"
#include <Eigen/Dense>
#pragma GCC diagnostic pop
#include <cassert>
#include <iostream>
#include <unordered_map>

namespace {
using namespace qcor::utils;
using namespace std::complex_literals;
Eigen::Matrix2cd getGateMat(const qop_t &in_op) {
  static const Eigen::Matrix2cd X_mat = []() {
    Eigen::Matrix2cd result = Eigen::MatrixXcd::Zero(2, 2);
    result << 0.0, 1.0, 1.0, 0.0;
    return result;
  }();
  static const Eigen::Matrix2cd Y_mat = []() {
    Eigen::Matrix2cd result = Eigen::MatrixXcd::Zero(2, 2);
    result << 0.0, -1i, 1i, 0.0;
    return result;
  }();
  static const Eigen::Matrix2cd Z_mat = []() {
    Eigen::Matrix2cd result = Eigen::MatrixXcd::Zero(2, 2);
    result << 1.0, 0.0, 0.0, -1.0;
    return result;
  }();
  static const Eigen::Matrix2cd H_mat = []() {
    Eigen::Matrix2cd result = Eigen::MatrixXcd::Zero(2, 2);
    result << M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, -M_SQRT1_2;
    return result;
  }();

  static const auto rx_mat = [](const std::vector<double> &in_params) {
    assert(in_params.size() == 1);
    auto theta = in_params[0];
    Eigen::Matrix2cd result = Eigen::MatrixXcd::Zero(2, 2);
    result << std::cos(0.5 * theta),
        std::complex<double>(0, -1) * std::sin(0.5 * theta),
        std::complex<double>(0, -1) * std::sin(0.5 * theta),
        std::cos(0.5 * theta);
    return result;
  };

  static const auto ry_mat = [](const std::vector<double> &in_params) {
    assert(in_params.size() == 1);
    auto theta = in_params[0];
    Eigen::Matrix2cd result = Eigen::MatrixXcd::Zero(2, 2);
    result << std::cos(0.5 * theta), -std::sin(0.5 * theta),
        std::sin(0.5 * theta), std::cos(0.5 * theta);
    return result;
  };

  static const auto rz_mat = [](const std::vector<double> &in_params) {
    assert(in_params.size() == 1);
    auto theta = in_params[0];
    Eigen::Matrix2cd result = Eigen::MatrixXcd::Zero(2, 2);
    result << std::exp(std::complex<double>(0, -0.5 * theta)), 0.0, 0.0,
        std::exp(std::complex<double>(0, 0.5 * theta));
    return result;
  };

  static const auto p_mat = [](const std::vector<double> &in_params) {
    assert(in_params.size() == 1);
    auto theta = in_params[0];
    Eigen::Matrix2cd result = Eigen::MatrixXcd::Zero(2, 2);
    result << 1.0, 0.0, 0.0, std::exp(std::complex<double>(0, theta));
    return result;
  };

  static const auto u3_mat = [](const std::vector<double> &in_params) {
    assert(in_params.size() == 3);
    auto in_theta = in_params[0];
    auto in_phi = in_params[1];
    auto in_lambda = in_params[2];

    Eigen::Matrix2cd result = Eigen::MatrixXcd::Zero(2, 2);

    // qpp::cmat gateMat(2, 2);
    result << std::cos(in_theta / 2.0),
        -std::exp(std::complex<double>(0, in_lambda)) *
            std::sin(in_theta / 2.0),
        std::exp(std::complex<double>(0, in_phi)) * std::sin(in_theta / 2.0),
        std::exp(std::complex<double>(0, in_phi + in_lambda)) *
            std::cos(in_theta / 2.0);

    return result;
  };

  static const std::unordered_map<std::string, Eigen::Matrix2cd>
      GateMatrixCache = {{"x", X_mat},           {"y", Y_mat},
                         {"z", Z_mat},           {"h", H_mat},
                         {"t", p_mat({M_PI_4})}, {"tdg", p_mat({-M_PI_4})},
                         {"s", p_mat({M_PI_2})}, {"sdg", p_mat({-M_PI_2})}};
  const auto &gateName = in_op.first;
  const auto &gateParams = in_op.second;
  const auto it = GateMatrixCache.find(gateName);
  if (it != GateMatrixCache.end()) {
    return it->second;
  }
  if (gateName == "rx") {
    return rx_mat(gateParams);
  }
  if (gateName == "ry") {
    return ry_mat(gateParams);
  }
  if (gateName == "rz") {
    return rz_mat(gateParams);
  }
  if (gateName == "u3") {
    return u3_mat(gateParams);
  }

  throw std::runtime_error("Unknown single qubit gate: " + gateName);
  return Eigen::MatrixXcd::Zero(2, 2);
}

// If the matrix is finite: no NaN elements
template <typename Derived>
inline bool isFinite(const Eigen::MatrixBase<Derived> &x) {
  return ((x - x).array() == (x - x).array()).all();
}

// Default tolerace for validation
constexpr double TOLERANCE = 1e-6;

template <typename Derived>
bool allClose(const Eigen::MatrixBase<Derived> &in_mat1,
              const Eigen::MatrixBase<Derived> &in_mat2,
              double in_tol = TOLERANCE) {
  if (!isFinite(in_mat1) || !isFinite(in_mat2)) {
    return false;
  }

  if (in_mat1.rows() == in_mat2.rows() && in_mat1.cols() == in_mat2.cols()) {
    for (int i = 0; i < in_mat1.rows(); ++i) {
      for (int j = 0; j < in_mat1.cols(); ++j) {
        if (std::abs(in_mat1(i, j) - in_mat2(i, j)) > in_tol) {
          return false;
        }
      }
    }

    return true;
  }
  return false;
}

// Use Z-Y decomposition of Nielsen and Chuang (Theorem 4.1).
// An arbitrary one qubit gate matrix can be writen as
// U = [ exp(j*(a-b/2-d/2))*cos(c/2), -exp(j*(a-b/2+d/2))*sin(c/2)
//       exp(j*(a+b/2-d/2))*sin(c/2), exp(j*(a+b/2+d/2))*cos(c/2)]
// where a,b,c,d are real numbers.
// Then U = exp(j*a) Rz(b) Ry(c) Rz(d).
std::tuple<double, double, double, double>
singleQubitGateDecompose(const Eigen::Matrix2cd &matrix) {
  static const Eigen::Matrix2cd ID_MAT = Eigen::Matrix2cd::Identity();
  if (allClose(matrix, ID_MAT)) {
    return std::make_tuple(0.0, 0.0, 0.0, 0.0);
  }
  const auto checkParams = [&matrix](double a, double bHalf, double cHalf,
                                     double dHalf) {
    Eigen::Matrix2cd U;
    U << std::exp(1i * (a - bHalf - dHalf)) * std::cos(cHalf),
        -std::exp(1i * (a - bHalf + dHalf)) * std::sin(cHalf),
        std::exp(1i * (a + bHalf - dHalf)) * std::sin(cHalf),
        std::exp(1i * (a + bHalf + dHalf)) * std::cos(cHalf);

    return allClose(U, matrix);
  };

  double a, bHalf, cHalf, dHalf;
  const double TOLERANCE = 1e-9;
  if (std::abs(matrix(0, 1)) < TOLERANCE) {
    auto two_a = fmod(std::arg(matrix(0, 0) * matrix(1, 1)), 2 * M_PI);
    a = (std::abs(two_a) < TOLERANCE || std::abs(two_a) > 2 * M_PI - TOLERANCE)
            ? 0
            : two_a / 2.0;
    auto dHalf = 0.0;
    auto b = std::arg(matrix(1, 1)) - std::arg(matrix(0, 0));
    std::vector<double> possibleBhalf{fmod(b / 2.0, 2 * M_PI),
                                      fmod(b / 2.0 + M_PI, 2.0 * M_PI)};
    std::vector<double> possibleChalf{0.0, M_PI};
    bool found = false;
    for (size_t i = 0; i < possibleBhalf.size(); ++i) {
      for (size_t j = 0; j < possibleChalf.size(); ++j) {
        bHalf = possibleBhalf[i];
        cHalf = possibleChalf[j];
        if (checkParams(a, bHalf, cHalf, dHalf)) {
          found = true;
          break;
        }
      }
      if (found) {
        break;
      }
    }
    assert(found);
  } else if (std::abs(matrix(0, 0)) < TOLERANCE) {
    auto two_a = fmod(std::arg(-matrix(0, 1) * matrix(1, 0)), 2 * M_PI);
    a = (std::abs(two_a) < TOLERANCE || std::abs(two_a) > 2 * M_PI - TOLERANCE)
            ? 0
            : two_a / 2.0;
    dHalf = 0;
    auto b = std::arg(matrix(1, 0)) - std::arg(matrix(0, 1)) + M_PI;
    std::vector<double> possibleBhalf{fmod(b / 2., 2 * M_PI),
                                      fmod(b / 2. + M_PI, 2 * M_PI)};
    std::vector<double> possibleChalf{M_PI / 2., 3. / 2. * M_PI};
    bool found = false;
    for (size_t i = 0; i < possibleBhalf.size(); ++i) {
      for (size_t j = 0; j < possibleChalf.size(); ++j) {
        bHalf = possibleBhalf[i];
        cHalf = possibleChalf[j];
        if (checkParams(a, bHalf, cHalf, dHalf)) {
          found = true;
          break;
        }
      }
      if (found) {
        break;
      }
    }
    assert(found);
  } else {
    auto two_a = fmod(std::arg(matrix(0, 0) * matrix(1, 1)), 2 * M_PI);
    a = (std::abs(two_a) < TOLERANCE || std::abs(two_a) > 2 * M_PI - TOLERANCE)
            ? 0
            : two_a / 2.0;
    auto two_d = 2. * std::arg(matrix(0, 1)) - 2. * std::arg(matrix(0, 0));
    std::vector<double> possibleDhalf{
        fmod(two_d / 4., 2 * M_PI), fmod(two_d / 4. + M_PI / 2., 2 * M_PI),
        fmod(two_d / 4. + M_PI, 2 * M_PI),
        fmod(two_d / 4. + 3. / 2. * M_PI, 2 * M_PI)};
    auto two_b = 2. * std::arg(matrix(1, 0)) - 2. * std::arg(matrix(0, 0));
    std::vector<double> possibleBhalf{
        fmod(two_b / 4., 2 * M_PI), fmod(two_b / 4. + M_PI / 2., 2 * M_PI),
        fmod(two_b / 4. + M_PI, 2 * M_PI),
        fmod(two_b / 4. + 3. / 2. * M_PI, 2 * M_PI)};
    auto tmp = std::acos(std::abs(matrix(1, 1)));
    std::vector<double> possibleChalf{
        fmod(tmp, 2 * M_PI), fmod(tmp + M_PI, 2 * M_PI),
        fmod(-1. * tmp, 2 * M_PI), fmod(-1. * tmp + M_PI, 2 * M_PI)};
    bool found = false;
    for (size_t i = 0; i < possibleBhalf.size(); ++i) {
      for (size_t j = 0; j < possibleChalf.size(); ++j) {
        for (size_t k = 0; k < possibleDhalf.size(); ++k) {
          bHalf = possibleBhalf[i];
          cHalf = possibleChalf[j];
          dHalf = possibleDhalf[k];
          if (checkParams(a, bHalf, cHalf, dHalf)) {
            found = true;
            break;
          }
        }
        if (found) {
          break;
        }
      }
      if (found) {
        break;
      }
    }
    assert(found);
  }

  // Final check:
  assert(checkParams(a, bHalf, cHalf, dHalf));
  return std::make_tuple(a, bHalf, cHalf, dHalf);
};

std::vector<pauli_decomp_t>
simplifySingleQubitSeq(double zAngleBefore, double yAngle, double zAngleAfter) {
  auto zExpBefore = zAngleBefore / M_PI - 0.5;
  auto middleExp = yAngle / M_PI;
  std::string middlePauli = "rx";
  auto zExpAfter = zAngleAfter / M_PI + 0.5;

  // Helper functions:
  const auto isNearZeroMod = [](double a, double period) -> bool {
    const auto halfPeriod = period / 2;
    const double TOL = 1e-8;
    return std::abs(fmod(a + halfPeriod, period) - halfPeriod) < TOL;
  };

  const auto toQuarterTurns = [](double in_exp) -> int {
    return static_cast<int>(round(2 * in_exp)) % 4;
  };

  const auto isCliffordRotation = [&](double in_exp) -> bool {
    return isNearZeroMod(in_exp, 0.5);
  };

  const auto isQuarterTurn = [&](double in_exp) -> bool {
    return (isCliffordRotation(in_exp) && toQuarterTurns(in_exp) % 2 == 1);
  };

  const auto isHalfTurn = [&](double in_exp) -> bool {
    return (isCliffordRotation(in_exp) && toQuarterTurns(in_exp) == 2);
  };

  const auto isNoTurn = [&](double in_exp) -> bool {
    return (isCliffordRotation(in_exp) && toQuarterTurns(in_exp) == 0);
  };

  // Clean up angles
  if (isCliffordRotation(zExpBefore)) {
    if ((isQuarterTurn(zExpBefore) || isQuarterTurn(zExpAfter)) !=
        (isHalfTurn(middleExp) && isNoTurn(zExpBefore - zExpAfter))) {
      zExpBefore += 0.5;
      zExpAfter -= 0.5;
      middlePauli = "ry";
    }
    if (isHalfTurn(zExpBefore) || isHalfTurn(zExpAfter)) {
      zExpBefore -= 1;
      zExpAfter += 1;
      middleExp = -middleExp;
    }
  }
  if (isNoTurn(middleExp)) {
    zExpBefore += zExpAfter;
    zExpAfter = 0;
  } else if (isHalfTurn(middleExp)) {
    zExpAfter -= zExpBefore;
    zExpBefore = 0;
  }

  std::vector<pauli_decomp_t> composite;
  if (!isNoTurn(zExpBefore)) {
    composite.emplace_back(std::make_pair("rz", zExpBefore * M_PI));
  }
  if (!isNoTurn(middleExp)) {
    composite.emplace_back(std::make_pair(middlePauli, middleExp * M_PI));
  }
  if (!isNoTurn(zExpAfter)) {
    composite.emplace_back(std::make_pair("rz", zExpAfter * M_PI));
  }
  return composite;
}

} // namespace
namespace qcor {
namespace utils {
std::vector<pauli_decomp_t>
decompose_gate_sequence(const std::vector<qop_t> &op_list) {
  Eigen::Matrix2cd totalU = Eigen::MatrixXcd::Identity(2, 2);
  for (const auto &op : op_list) {
    // std::cout << "Gate: " << op.first << ": " << op.second.size() << "\n";
    totalU = getGateMat(op) * totalU;
  }

  // std::cout << "Total U = " << totalU << "\n";
  auto [a, bHalf, cHalf, dHalf] = singleQubitGateDecompose(totalU);

  // Validate U = exp(j*a) Rz(b) Ry(c) Rz(d).
  const auto validate = [](const Eigen::Matrix2cd &in_mat, double a, double b,
                           double c, double d) {
    Eigen::Matrix2cd Rz_b, Ry_c, Rz_d;
    Rz_b << std::exp(-1i * b / 2.0), 0, 0, std::exp(1i * b / 2.0);
    Rz_d << std::exp(-1i * d / 2.0), 0, 0, std::exp(1i * d / 2.0);
    Ry_c << std::cos(c / 2), -std::sin(c / 2), std::sin(c / 2), std::cos(c / 2);
    Eigen::Matrix2cd mat = std::exp(1i * a) * Rz_b * Ry_c * Rz_d;
    return allClose(in_mat, mat);
  };
  // Validate the *raw* decomposition
  assert(validate(totalU, a, 2 * bHalf, 2 * cHalf, 2 * dHalf));
  return simplifySingleQubitSeq(2 * dHalf, 2 * cHalf, 2 * bHalf);
}
} // namespace utils
} // namespace qcor
