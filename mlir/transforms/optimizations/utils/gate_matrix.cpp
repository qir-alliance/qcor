#include "gate_matrix.hpp"
#include <Eigen/Dense>
#include <unordered_map>
#include <cassert>
#include <iostream>

namespace {
using namespace qcor::mlir::utils;
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

  static const std::unordered_map<std::string, Eigen::Matrix2cd>
      GateMatrixCache = {{"x", X_mat},           {"y", Y_mat},
                         {"z", Z_mat},           {"h", H_mat},
                         {"t", p_mat({M_PI_4})}, {"tdg", p_mat({-M_PI_4})},
                         {"s", p_mat({M_PI_2})}, {"sdg", p_mat({-M_PI_2})}};
  const auto &gateName = in_op.first;
  const auto &gateParams = in_op.second;
  const auto it = GateMatrixCache.find(gateName);
  if (it != GateMatrixCache.end()) {
    it->second;
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

  
  return Eigen::MatrixXcd::Zero(2, 2);
}
} // namespace
namespace qcor {
namespace mlir {
namespace utils {
std::vector<pauli_decomp_t>
decompose_gate_sequence(const std::vector<qop_t> &op_list) {
  Eigen::Matrix2cd totalU = Eigen::MatrixXcd::Identity(2, 2);
  for (const auto& op: op_list) {
    totalU = getGateMat(op) * totalU;
  }

  std::cout << "Total U = " << totalU << "\n";

  // TODO:
  return {};
}
} // namespace utils
} // namespace mlir
} // namespace qcor
