#include "qcor.hpp"

// Quantum Fourier Transform
__qpu__ void qft(qreg q) {
  // Local Declarations
  const auto nQubits = q.size();
  for (int qIdx = 0; qIdx < nQubits; ++qIdx) {
    H(q[qIdx]);
    for (int j = qIdx + 1; j < nQubits; ++j) {
      const double theta = M_PI/std::pow(2.0, j - qIdx);
      CPhase(q[j], q[qIdx], theta);
    }
  }

  // Swap qubits
  for (int qIdx = 0; qIdx < nQubits / 2; ++qIdx) {
    Swap(q[qIdx], q[nQubits - qIdx - 1]);
  }
}

// Inverse Quantum Fourier Transform
__qpu__ void iqft(qreg q) {
  // Local Declarations
  const auto nQubits = q.size();
  // Swap qubits
  const int startIdx = nQubits / 2 - 1;
  for (int qIdx = startIdx; qIdx >= 0; --qIdx) {
    Swap(q[qIdx], q[nQubits - qIdx - 1]);
  }
    
  for (int qIdx = nQubits - 1; qIdx >= 0; --qIdx) {
    for (int j = nQubits - 1; j > qIdx; --j) {
      const double theta = -M_PI/std::pow(2.0, j - qIdx);
      CPhase(q[j], q[qIdx], theta);
    }
    H(q[qIdx]);
  }
}

__qpu__ void f(qreg q) {
  const auto nQubits = q.size();
  // TODO: check if this is possible
  qft(q);
  iqft(q);
  for (int qIdx = 0; qIdx < nQubits / 2; ++qIdx) {
    Measure(q[qIdx]);
  }
}


int main(int argc, char **argv) {
  // Allocate 4 qubits
  auto q = qalloc(4);
  f(q);
  // dump the results
  q.print();
}
