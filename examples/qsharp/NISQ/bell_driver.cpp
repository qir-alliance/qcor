#include <iostream> 
#include <vector>
#include "import_kernel_utils.hpp"

// Util pre-processor to wrap Q# operation 
// in a QCOR QuantumKernel.
qcor_import_qsharp_kernel(QCOR__Bell);

int main() {
  // Allocate 2 qubits
  auto q = qalloc(2);
  QCOR__Bell(q);
  q.print();
  return 0;
}