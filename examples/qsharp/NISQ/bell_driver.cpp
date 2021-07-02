#include <iostream> 
#include <vector>
#include "import_kernel_utils.hpp"

// Util pre-processor to wrap Q# operation 
// in a QCOR QuantumKernel.
// Compile:
// Note: need to use alpha package since this kernel will take a qubit array.
// qcor -qdk-version 0.17.2106148041-alpha bell.qs bell_driver.cpp -shots 1024
qcor_import_qsharp_kernel(QCOR__Bell);

int main() {
  // Allocate 2 qubits
  auto q = qalloc(2);
  QCOR__Bell(q);
  q.print();
  return 0;
}