#include <iostream> 
#include <vector>
#include "qcor.hpp"
#include "import_kernel_utils.hpp"

using Qubit = uint64_t;
using QReg = std::vector<Qubit*>;
// Util pre-processor to wrap Q# operation in a QCOR QuantumKernel.
qcor_import_qsharp_kernel(QCOR__TestKernel, double);

// Compile with:
// Include both the qsharp source and this driver file
// in the command line.
// $ qcor -qpu aer:ibmqx2 -shots 1024 kernel_nisq.qs kernel_nisq_driver.cpp
// Run with:
// $ ./a.out
int main() {
  auto q = qalloc(3);
  qcor::set_verbose(true);
  // QCOR__TestKernel(q, 1.0);
  // q.print();

  // Integrate w/ QCOR's kernel utility...
  // e.g. kernel print-out...
  std::cout << "HELLO:\n";
  QCOR__TestKernel::print_kernel(std::cout, q, M_PI/4);

  return 0;
}