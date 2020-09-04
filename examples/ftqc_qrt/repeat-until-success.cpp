#include <qalloc>
// Compile with: qcor -qpu qpp -qrt ftqc repeat-until-success.cpp
// Execute: ./a.out
// We should get the print out conditioned by the measurement.
// If not using the "ftqc" QRT, this will cause errors since the Measure results
// are not available yet.

__qpu__ void rus(qreg q, int maxIter) {
  using qcor::xasm;
  // Note: control = q[0], eigenstate = q[1]
  // U = Z; P = X
  bool measuredZero = false;
  bool measuredOne = false;
  X(q[1]);
  for (int i = 0; i < maxIter; ++i) {
    std::cout << "Iter: " << i << "\n";
    H(q[0]);
    CZ(q[0], q[1]);
    H(q[0]);
    Measure(q[0]);
    if (q.cReg(0)) {
      // Fix up: if measure = 1, reset the control qubit back to 0.
      // Note: the eigenstate qubit remains coherent during this RUS loop.
      X(q[0]);
    }

    measuredZero = measuredZero || (q.cReg(0) == false);
    measuredOne = measuredOne || (q.cReg(0) == true);
    if (measuredZero && measuredOne) {
      std::cout << "Success after " << i << " iterations.\n";
      break;
    }
  }

  if (!measuredZero || !measuredOne) {
    std::cout << "Failed!!!\n";
  }
}

int main() {
  auto q = qalloc(2);
  rus(q, 100);
}
