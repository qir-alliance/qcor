#pragma once
#include <tuple>
namespace qcor {
namespace utils {
// Generalized rotation (3 angles)
// rz - sx - rz - sx - rz
using GenRot_t = std::tuple<double, double, double>;
enum class PauliLabel { I, X, Y, Z };
// Computes z rotation angles in a randomized Pauli frame.
// - in_pauli: the randomined Pauli op
GenRot_t computeRotationInPauliFrame(const GenRot_t &in_rot,
                                     PauliLabel in_newPauli,
                                     PauliLabel &io_netPauli);
} // namespace utils
} // namespace qcor