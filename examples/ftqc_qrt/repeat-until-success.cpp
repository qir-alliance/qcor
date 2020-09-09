#include <qalloc>
// Compile with: qcor -qpu qpp -qrt ftqc repeat-until-success.cpp
// Execute: ./a.out
// We should get the print out conditioned by the measurement.
// If not using the "ftqc" QRT, this will cause errors since the Measure results
// are not available yet.

// Using Repeat-Until-Success pattern to prepare a quantum state.
// https://docs.microsoft.com/en-us/quantum/user-guide/using-qsharp/control-flow#rus-to-prepare-a-quantum-state
__qpu__ void PrepareStateUsingRUS(qreg q, int maxIter) {
  using qcor::xasm;
  // Note: target = q[0], aux = q[1]
  H(q[1]);
  // We limit the max number of RUS iterations.
  for (int i = 0; i < maxIter; ++i) {
    std::cout << "Iter: " << i << "\n";
    Tdg(q[1]);
    CNOT(q[0], q[1]);
    T(q[1]);

    // In order to measure in the PauliX basis, changes the basis.
    H(q[1]);
    if (!Measure(q[1])) {
      // Success (until (outcome == Zero))
      std::cout << "Success after " << i + 1 << " iterations.\n";
      break;
    } 
    else {
      // Measure 1: |1> state
      // Fix up: Bring the auxiliary and target qubits back to |+> state.
      X(q[1]);
      H(q[1]);
      X(q[0]);
      H(q[0]);
    }
  }
}

int main() {
  // qcor::set_verbose(true);
  auto q = qalloc(2);
  PrepareStateUsingRUS(q, 100);
}
