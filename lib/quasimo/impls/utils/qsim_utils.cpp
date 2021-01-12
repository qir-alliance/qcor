#include "qsim_utils.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <cassert>
namespace qcor {
namespace QuaSiMo {
std::shared_ptr<CostFunctionEvaluator>
getEvaluator(Observable *observable, const HeterogeneousMap &params) {
  // If an evaluator was provided explicitly:
  if (params.pointerLikeExists<CostFunctionEvaluator>("evaluator")) {
    return xacc::as_shared_ptr(
        params.getPointerLike<CostFunctionEvaluator>("evaluator"));
  }

  // Cost Evaluator was provided by name:
  if (params.stringExists("evaluator")) {
    return getObjEvaluator(observable, params.getString("evaluator"));
  }

  // No specific evaluator/evaluation method was requested,
  // use the default one (partial tomography based).
  return getObjEvaluator(observable);
}

PronyResult pronyFit(const std::vector<std::complex<double>> &in_signal) {
  assert(!in_signal.empty());

  // Returns the Hankel matrix constructed from the first column c, and
  // (optionally) the last row r.
  auto hankelMat =
      [](const std::vector<std::complex<double>> &c,
         const std::vector<std::complex<double>> &r) -> Eigen::MatrixXcd {
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(c.size(), r.size());
    const auto m = c.size();
    for (int i = 0; i < result.rows(); ++i) {
      for (int j = 0; j < result.cols(); ++j) {
        // H(i,j) = c(i+j),  i+j < m;
        if (i + j < m) {
          result(i, j) = c[i + j];
        } else {
          // H(i,j) = r(i+j+1-m),  otherwise
          result(i, j) = r[i + j + 1 - m];
        }
      }
    }

    return result;
  };

  // Pythonic vector slice:
  auto vectorSlice = [](const std::vector<std::complex<double>> &in_vec,
                        int in_start, int in_end) {
    std::vector<std::complex<double>> result;
    auto startIter = in_vec.begin() + in_start;
    auto endIter =
        in_end > 0 ? in_vec.begin() + in_end : (in_vec.end() + in_end);
    result.assign(startIter, endIter);
    return result;
  };

  const size_t num_freqs = in_signal.size() / 2;
  auto hankel0 = hankelMat(vectorSlice(in_signal, 0, num_freqs),
                           vectorSlice(in_signal, num_freqs - 1, -1));
  auto hankel1 = hankelMat(vectorSlice(in_signal, 1, num_freqs + 1),
                           vectorSlice(in_signal, num_freqs, 0));
  // std::cout << "Hankel Matrix 0: \n" << hankel0 << "\n";
  // std::cout << "Hankel Matrix 1: \n" << hankel1 << "\n";

  hankel0.transposeInPlace();
  hankel1.transposeInPlace();
  Eigen::MatrixXcd shift_matrix =
      Eigen::MatrixXcd::Zero(hankel0.cols(), hankel1.cols());

  for (size_t i = 0; i < hankel1.cols(); ++i) {
    Eigen::VectorXcd shift_matrix_col =
        hankel0.fullPivHouseholderQr().solve(hankel1.col(i));
    shift_matrix.col(i) = shift_matrix_col;
  }
  // std::cout << "Shift matrix: \n" << shift_matrix << "\n";

  shift_matrix.transposeInPlace();
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> s(shift_matrix);
  auto phases = s.eigenvalues();
  Eigen::MatrixXcd generation_matrix =
      Eigen::MatrixXcd::Zero(in_signal.size(), phases.size());
  for (int i = 0; i < generation_matrix.rows(); ++i) {
    for (int j = 0; j < generation_matrix.cols(); ++j) {
      generation_matrix(i, j) = std::pow(phases[j], i);
    }
  }

  auto signalData = in_signal;
  Eigen::VectorXcd signal =
      Eigen::Map<Eigen::VectorXcd>(signalData.data(), signalData.size());
  Eigen::VectorXcd amplitudes =
      generation_matrix.fullPivHouseholderQr().solve(signal);
  assert(phases.size() == amplitudes.size());
  // std::cout << "Amplitude:\n" << amplitudes << "\n";
  // std::cout << "Phases:\n" << phases << "\n";
  PronyResult finalResult;
  for (size_t i = 0; i < phases.size(); ++i) {
    finalResult.emplace_back(std::make_pair(amplitudes(i), phases(i)));
  }

  // Sort by amplitude:
  std::sort(finalResult.begin(), finalResult.end(),
            [](const auto &a, const auto &b) {
              return std::abs(a.first) < std::abs(b.first);
            });

  return finalResult;
}
} // namespace QuaSiMo
} // namespace qcor