#include <iostream>
#include <vector>

// For testing Q# IQFT circuit...
// Currently, Qubits are **not** allowed in EntryPoint => need to use a dummy entry point.
qcor_include_qsharp(QCOR__Dummy, void);

// Compile with:
// Include both the qsharp source and this driver file
// in the command line.
// Use print-final-submission to see the instructions executed.
// $ qcor -qrt ftqc -print-final-submission ...
// Run with:
// $ ./a.out
int main() {
  QCOR__Dummy();
  return 0;
}