#include <qalloc>
// Compile with: qcor -qpu qpp -qrt ftqc simple-demo.cpp
// Execute: ./a.out
// We should get the print out conditioned by the measurement.
// If not using the "ftqc" QRT, this will cause errors since the Measure results
// are not available yet.

__qpu__ void bell(qreg q, int nbRuns) {
  using qcor::xasm;
  for (int i = 0; i < nbRuns; ++i) {
    H(q[0]);
    CX(q[0], q[1]);
    Measure(q[0]);
    Measure(q[1]);
    if (q.cReg(0) == q.cReg(1)) {
      std::cout << "Iter " << i << ": Matched!\n";
    } else {
      // Should only happen if using a real (noisy) backend.
      std::cout << "Iter " << i << ": Not matched!\n";
    }
    // Reset qubits
    if (q.cReg(0)) {
      X(q[0]);
    }
    if (q.cReg(1)) {
      X(q[1]);
    }
  }
}

int main() {
  auto q = qalloc(2);
  bell(q, 100);
}
