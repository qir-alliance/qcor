#include "clifford_gate_utils.hpp"
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <set>

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
                                     PauliLabel in_netPauli) {
  auto [theta1, theta2, theta3] = in_rot;

  theta1 = mod_2pi(theta1);
  theta2 = mod_2pi(theta2);
  theta3 = mod_2pi(theta3);
  if (in_netPauli == PauliLabel::X || in_netPauli == PauliLabel::Z) {
    theta2 *= -1.0;
  }
  if (in_netPauli == PauliLabel::X || in_netPauli == PauliLabel::Y) {
    theta3 *= -1.0;
    theta1 *= -1.0;
  }

  // if x or y
  if (in_newPauli == PauliLabel::X || in_newPauli == PauliLabel::Y) {
    theta1 = -theta1 + M_PI;
    theta2 = theta2 + M_PI;
  }
  // if y or z
  if (in_newPauli == PauliLabel::Y || in_newPauli == PauliLabel::Z) {
    theta1 = theta1 + M_PI;
  }

  // make everything between - pi and pi
  theta1 = mod_2pi(theta1);
  theta2 = mod_2pi(theta2);
  theta3 = mod_2pi(theta3);
  return std::make_tuple(theta1, theta2, theta3);
}

GenRot_t invU3Gate(const GenRot_t &in_rot) {
  auto [theta1, theta2, theta3] = in_rot;
  theta1 = mod_2pi(M_PI - theta1);
  theta2 = mod_2pi(-theta2);
  theta3 = mod_2pi(-theta3 + M_PI);
  return std::make_tuple(theta1, theta2, theta3);
}

SrepDict_t computeGateSymplecticRepresentations(
    const std::vector<std::string> &in_gateList) {
  static const SrepDict_t standardDict = []() {
    std::unordered_map<std::string, Smatrix_t> complete_s_dict;
    std::unordered_map<std::string, Pvec_t> complete_p_dict;

    // The Pauli gates
    complete_s_dict["I"] = std::vector<std::vector<int>>{{1, 0}, {0, 1}};
    complete_s_dict["X"] = std::vector<std::vector<int>>{{1, 0}, {0, 1}};
    complete_s_dict["Y"] = std::vector<std::vector<int>>{{1, 0}, {0, 1}};
    complete_s_dict["Z"] = std::vector<std::vector<int>>{{1, 0}, {0, 1}};

    complete_p_dict["I"] = std::vector<int>{0, 0};
    complete_p_dict["X"] = std::vector<int>{0, 2};
    complete_p_dict["Y"] = std::vector<int>{2, 2};
    complete_p_dict["Z"] = std::vector<int>{2, 0};

    // Five single qubit gates that each represent one of five classes of
    // Cliffords that equivalent up to Pauli gates and are not equivalent to
    // idle (that class is covered by any one of the Pauli gates above).
    complete_s_dict["H"] = std::vector<std::vector<int>>{{0, 1}, {1, 0}};
    complete_s_dict["P"] = std::vector<std::vector<int>>{{1, 0}, {1, 1}};
    complete_s_dict["PH"] = std::vector<std::vector<int>>{{0, 1}, {1, 1}};
    complete_s_dict["HP"] = std::vector<std::vector<int>>{{1, 1}, {1, 0}};
    complete_s_dict["HPH"] = std::vector<std::vector<int>>{{1, 1}, {0, 1}};
    complete_p_dict["H"] = std::vector<int>{0, 0};
    complete_p_dict["P"] = std::vector<int>{1, 0};
    complete_p_dict["PH"] = std::vector<int>{0, 1};
    complete_p_dict["HP"] = std::vector<int>{3, 0};
    complete_p_dict["HPH"] = std::vector<int>{0, 3};
    // The full 1-qubit Cliffor group, using the same labelling as in
    // extras.rb.group
    complete_s_dict["C0"] = std::vector<std::vector<int>>{{1, 0}, {0, 1}};
    complete_p_dict["C0"] = std::vector<int>{0, 0};
    complete_s_dict["C1"] = std::vector<std::vector<int>>{{1, 1}, {1, 0}};
    complete_p_dict["C1"] = std::vector<int>{1, 0};
    complete_s_dict["C2"] = std::vector<std::vector<int>>{{0, 1}, {1, 1}};
    complete_p_dict["C2"] = std::vector<int>{0, 1};
    complete_s_dict["C3"] = std::vector<std::vector<int>>{{1, 0}, {0, 1}};
    complete_p_dict["C3"] = std::vector<int>{0, 2};
    complete_s_dict["C4"] = std::vector<std::vector<int>>{{1, 1}, {1, 0}};
    complete_p_dict["C4"] = std::vector<int>{1, 2};
    complete_s_dict["C5"] = std::vector<std::vector<int>>{{0, 1}, {1, 1}};
    complete_p_dict["C5"] = std::vector<int>{0, 3};
    complete_s_dict["C6"] = std::vector<std::vector<int>>{{1, 0}, {0, 1}};
    complete_p_dict["C6"] = std::vector<int>{2, 2};
    complete_s_dict["C7"] = std::vector<std::vector<int>>{{1, 1}, {1, 0}};
    complete_p_dict["C7"] = std::vector<int>{3, 2};
    complete_s_dict["C8"] = std::vector<std::vector<int>>{{0, 1}, {1, 1}};
    complete_p_dict["C8"] = std::vector<int>{2, 3};
    complete_s_dict["C9"] = std::vector<std::vector<int>>{{1, 0}, {0, 1}};
    complete_p_dict["C9"] = std::vector<int>{2, 0};
    complete_s_dict["C10"] = std::vector<std::vector<int>>{{1, 1}, {1, 0}};
    complete_p_dict["C10"] = std::vector<int>{3, 0};
    complete_s_dict["C11"] = std::vector<std::vector<int>>{{0, 1}, {1, 1}};
    complete_p_dict["C11"] = std::vector<int>{2, 1};
    complete_s_dict["C12"] = std::vector<std::vector<int>>{{0, 1}, {1, 0}};
    complete_p_dict["C12"] = std::vector<int>{0, 0};
    complete_s_dict["C13"] = std::vector<std::vector<int>>{{1, 1}, {0, 1}};
    complete_p_dict["C13"] = std::vector<int>{0, 1};
    complete_s_dict["C14"] = std::vector<std::vector<int>>{{1, 0}, {1, 1}};
    complete_p_dict["C14"] = std::vector<int>{1, 0};
    complete_s_dict["C15"] = std::vector<std::vector<int>>{{0, 1}, {1, 0}};
    complete_p_dict["C15"] = std::vector<int>{0, 2};
    complete_s_dict["C16"] = std::vector<std::vector<int>>{{1, 1}, {0, 1}};
    complete_p_dict["C16"] = std::vector<int>{0, 3};
    complete_s_dict["C17"] = std::vector<std::vector<int>>{{1, 0}, {1, 1}};
    complete_p_dict["C17"] = std::vector<int>{1, 2};
    complete_s_dict["C18"] = std::vector<std::vector<int>>{{0, 1}, {1, 0}};
    complete_p_dict["C18"] = std::vector<int>{2, 2};
    complete_s_dict["C19"] = std::vector<std::vector<int>>{{1, 1}, {0, 1}};
    complete_p_dict["C19"] = std::vector<int>{2, 3};
    complete_s_dict["C20"] = std::vector<std::vector<int>>{{1, 0}, {1, 1}};
    complete_p_dict["C20"] = std::vector<int>{3, 2};
    complete_s_dict["C21"] = std::vector<std::vector<int>>{{0, 1}, {1, 0}};
    complete_p_dict["C21"] = std::vector<int>{2, 0};
    complete_s_dict["C22"] = std::vector<std::vector<int>>{{1, 1}, {0, 1}};
    complete_p_dict["C22"] = std::vector<int>{2, 1};
    complete_s_dict["C23"] = std::vector<std::vector<int>>{{1, 0}, {1, 1}};
    complete_p_dict["C23"] = std::vector<int>{3, 0};
    // The CNOT gate, CPHASE gate, and SWAP gate.
    complete_s_dict["CNOT"] = std::vector<std::vector<int>>{
        {1, 0, 0, 0}, {1, 1, 0, 0}, {0, 0, 1, 1}, {0, 0, 0, 1}};
    complete_s_dict["CPHASE"] = std::vector<std::vector<int>>{
        {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 1, 1, 0}, {1, 0, 0, 1}};
    complete_s_dict["SWAP"] = std::vector<std::vector<int>>{
        {0, 1, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
    complete_p_dict["CNOT"] = std::vector<int>{0, 0, 0, 0};
    complete_p_dict["CPHASE"] = std::vector<int>{0, 0, 0, 0};
    complete_p_dict["SWAP"] = std::vector<int>{0, 0, 0, 0};

    SrepDict_t result;
    for (const auto &[k, v] : complete_s_dict) {
      assert(complete_p_dict.find(k) != complete_p_dict.end());
      result[k] = std::make_pair(v, complete_p_dict[k]);
    }
    return result;
  }();

  if (!in_gateList.empty()) {
    // Filter a subset of gates:
    SrepDict_t result;
    for (const auto &gateLabel : in_gateList) {
      const auto iter = standardDict.find(gateLabel);
      assert(iter != standardDict.end());
      result[gateLabel] = iter->second;
    }
    return result;
  }
  return standardDict;
}

Srep_t computeLayerSymplecticRepresentations(const CliffordGateLayer_t &layers,
                                             int nQubits,
                                             const SrepDict_t &srep_dict) {
  // Initilize
  Pvec_t p(2 * nQubits, 0);
  Smatrix_t s(2 * nQubits, p);
  std::set<int> seen_qubits;
  for (const auto &[name, operands] : layers) {
    const auto iter = srep_dict.find(name);
    assert(iter != srep_dict.end());
    const auto &[matrix, phase] = iter->second;
    const auto nforgate = operands.size();
    assert(nforgate > 0);
    for (int ind1 = 0; ind1 < operands.size(); ++ind1) {
      const auto qindex1 = operands[ind1];
      assert(seen_qubits.find(qindex1) == seen_qubits.end());
      seen_qubits.emplace(qindex1);
      for (int ind2 = 0; ind2 < operands.size(); ++ind2) {
        const auto qindex2 = operands[ind2];
        // Put in the symp matrix elements
        s[qindex1][qindex2] = matrix[ind1][ind2];
        s[qindex1][qindex2 + nQubits] = matrix[ind1][ind2 + nforgate];
        s[qindex1 + nQubits][qindex2] = matrix[ind1 + nforgate][ind2];
        s[qindex1 + nQubits][qindex2 + nQubits] =
            matrix[ind1 + nforgate][ind2 + nforgate];
      }
      // Put in the phase elements
      p[qindex1] = phase[ind1];
      p[qindex1 + nQubits] = phase[ind1 + nforgate];
    }
  }

  return std::make_pair(s, p);
}

Srep_t composeCliffords(const Srep_t &C1, const Srep_t &C2) {
  assert(C1.first.size() == C2.first.size());
  assert(C1.second.size() == C2.second.size());
  assert(C1.first.size() % 2 == 0);
  const int n = C1.first.size() / 2;
  using Mat_t = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
  using Vec_t = Eigen::Matrix<int, Eigen::Dynamic, 1>;
  Mat_t s1(2 * n, 2 * n);
  Mat_t s2(2 * n, 2 * n);
  Vec_t p1(2 * n);
  Vec_t p2(2 * n);
  // Load to eigen for processing
  for (int i = 0; i < 2 * n; ++i) {
    for (int j = 0; j < 2 * n; ++j) {
      s1(i, j) = C1.first[i][j];
      s2(i, j) = C2.first[i][j];
    }
    p1(i) = C1.second[i];
    p2(i) = C2.second[i];
  }

  Mat_t s = s2 * s1;
  // Mod 2
  for (int i = 0; i < 2 * n; ++i) {
    for (int j = 0; j < 2 * n; ++j) {
      s(i, j) = s(i, j) % 2;
    }
  }

  Mat_t u = Mat_t::Zero(2 * n, 2 * n);
  u(Eigen::seq(n, 2 * n - 1), Eigen::seq(0, n - 1)) = Mat_t::Identity(n, n);

  Vec_t vec1 = s1.transpose() * p2;
  Mat_t inner = (s2.transpose() * u) * s2;
  const auto strictly_upper_triangle = [](const Mat_t &m) -> Mat_t {
    auto l = m.rows();
    Mat_t out = m;

    for (int i = 0; i < l; ++i) {
      for (int j = 0; j < i + 1; ++j) {
        out(i, j) = 0;
      }
    }
    return out;
  };

  // Returns a diagonal matrix containing the diagonal of m.
  const auto diagonal_as_matrix = [](const Mat_t &m) -> Mat_t {
    auto l = m.rows();
    Mat_t out = Mat_t::Zero(l, l);
    for (int i = 0; i < l; ++i) {
      out(i, i) = m(i, i);
    }

    return out;
  };

  // Returns a 1D array containing the diagonal of the input square 2D array m.
  const auto diagonal_as_vec = [](const Mat_t &m) -> Mat_t {
    auto l = m.rows();
    Vec_t vec = Vec_t::Zero(l);
    for (int i = 0; i < l; ++i) {
      vec(i) = m(i, i);
    }

    return vec;
  };

  Mat_t matrix = 2 * strictly_upper_triangle(inner) + diagonal_as_matrix(inner);
  Vec_t vec2 = diagonal_as_vec((s1.transpose() * matrix) * s1);
  Vec_t vec3 = s1.transpose() * diagonal_as_vec(inner);
  Vec_t p = p1 + vec1 + vec2 - vec3;
  for (int i = 0; i < p.size(); ++i) {
    p(i) = p(i) % 4;
  }

  Pvec_t p_res(2 * n, 0);
  Smatrix_t s_res(2 * n, p_res);
  for (int i = 0; i < 2 * n; ++i) {
    for (int j = 0; j < 2 * n; ++j) {
      s_res[i][j] = s(i, j);
    }
    p_res[i] = p(i);
  }
  return std::make_pair(s_res, p_res);
}

Srep_t computeCircuitSymplecticRepresentations(
    const std::vector<CliffordGateLayer_t> &layers, int nQubits,
    const SrepDict_t &srep_dict) {
  // Initilize
  Pvec_t p(2 * nQubits, 0);
  Smatrix_t s(2 * nQubits, p);
  // S must be initialized as an identity matrix
  for (int i = 0; i < 2 * nQubits; ++i) {
    s[i][i] = 1;
  }
  for (const auto &layer : layers) {
    const auto layerRep =
        computeLayerSymplecticRepresentations(layer, nQubits, srep_dict);

    std::tie(s, p) = composeCliffords(std::make_pair(s, p), layerRep);
  }
  return std::make_pair(s, p);
}

std::vector<PauliLabel> find_pauli_labels(const Pvec_t &pvec) {
  assert(pvec.size() % 2 == 0);
  const auto n = pvec.size() / 2;
  std::vector<int> v(n, 0);
  for (int i = 0; i < n; ++i) {
    v[i] = (pvec[i] / 2) + 2 * (pvec[n + i] / 2);
  }
  // [0,0]=I, [2,0]=Z, [0,2]=X, and [2,2]=Y.
  std::vector<PauliLabel> result;
  for (const auto &el : v) {
    assert(el < 4);
    static const std::vector<PauliLabel> ARRAY{PauliLabel::I, PauliLabel::Z,
                                               PauliLabel::X, PauliLabel::Y};
    result.emplace_back(ARRAY[el]);
  }
  return result;
}
} // namespace utils
} // namespace qcor