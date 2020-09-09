#include <qalloc>
// Compile with: qcor -qpu qpp -qrt ftqc simple-demo.cpp
// or with noise: qcor -qpu aer[noise-model:noise_model.json] -qrt ftqc simple-demo.cpp
// Execute: ./a.out
// We should get the print out conditioned by the measurement.
// If not using the "ftqc" QRT, this will cause errors since the Measure results
// are not available yet.
__qpu__ void bell(qreg q, int nbRuns) {
  using qcor::xasm;
  for (int i = 0; i < nbRuns; ++i) {
    H(q[0]);
    CX(q[0], q[1]);
    const bool q0Result = Measure(q[0]);
    const bool q1Result = Measure(q[1]);
    if (q0Result == q1Result) {
      std::cout << "Iter " << i << ": Matched!\n";
    } else {
      std::cout << "Iter " << i << ": NOT Matched!\n";
    }
    // Reset qubits
    if (q0Result) {
      X(q[0]);
    }
    if (q1Result) {
      X(q[1]);
    }
  }
}

int main() {
  auto q = qalloc(2);
  bell(q, 100);
}
