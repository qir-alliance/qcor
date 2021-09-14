#pragma once
#include <tuple>
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>
namespace qcor {
namespace utils {
// Generalized rotation (3 angles)
// rz - sx - rz - sx - rz
using GenRot_t = std::tuple<double, double, double>;
enum class PauliLabel { I, X, Y, Z };
static inline std::vector<PauliLabel>
    ALL_PAULI_OPS({PauliLabel::I, PauliLabel::X, PauliLabel::Y, PauliLabel::Z});
inline std::ostream &operator<<(std::ostream &out, PauliLabel value) {
  std::string s;
#define PROCESS_VAL(p)                                                         \
  case (p):                                                                    \
    s = #p;                                                                    \
    break;
  switch (value) {
    PROCESS_VAL(PauliLabel::I);
    PROCESS_VAL(PauliLabel::X);
    PROCESS_VAL(PauliLabel::Y);
    PROCESS_VAL(PauliLabel::Z);
  }
#undef PROCESS_VAL

  return out << s.back();
}

// Symplectic matrix and phase vector representations
using Smatrix_t = std::vector<std::vector<int>>;
using Pvec_t = std::vector<int>;
using Srep_t = std::pair<Smatrix_t, Pvec_t>;
using SrepDict_t = std::unordered_map<std::string, Srep_t>;
// List of gate label and operands
using CliffordGateLayer_t =
    std::vector<std::pair<std::string, std::vector<int>>>;
// Computes z rotation angles in a randomized Pauli frame.
// - in_pauli: the randomined Pauli op
GenRot_t computeRotationInPauliFrame(const GenRot_t &in_rot,
                                     PauliLabel in_newPauli,
                                     PauliLabel in_netPauli);
// makes a compiled version of the inverse of a compiled general unitary
// negate angles for inverse based on central pauli, account for recompiling the
// X(-pi/2) into X(pi/2)
GenRot_t invU3Gate(const GenRot_t &in_rot);
// Creates a dictionary of the symplectic representations of
// Clifford gates.
// Returns a dictionary of (s matrix, phase vector) pairs,
// i.e., symplectic matrix and phase vector representing the operation label
// given by the key.
SrepDict_t computeGateSymplecticRepresentations(
    const std::vector<std::string> &in_gateList = {});

Srep_t computeLayerSymplecticRepresentations(const CliffordGateLayer_t &layers,
                                             int nQubits,
                                             const SrepDict_t &srep_dict);

Srep_t computeCircuitSymplecticRepresentations(
    const std::vector<CliffordGateLayer_t> &layers, int nQubits,
    const SrepDict_t &srep_dict);

// Multiplies two cliffords in the symplectic representation.
// C2 times C1 (i.e., C1 acts first)
Srep_t composeCliffords(const Srep_t &C1, const Srep_t &C2);

std::vector<PauliLabel> find_pauli_labels(const Pvec_t& pvec);
} // namespace utils
} // namespace qcor