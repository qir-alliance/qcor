// Demonstrate loading openqasm file wiht 
// #include call. Also demonstrate the 
// qcor::measure_all API call. 
// run this with 
// qcor -qrt -qpu aer grover.cpp
// ./a.out

#include "qcor.hpp"

// Mandate that for .qasm file if there
// exists N qreg allocation calls, then we
// have to pass N qcor qreg vars to the function
__qpu__ void grover_5(qreg q) {
  using qcor::openqasm;

#include "grover_5.qasm"

}

int main() {

  // Allocate the qubits
  auto q = qalloc(9);

  // This kernel is unmeasured, 
  // add measures to them
  auto measure_grov = qcor::measure_all(grover_5, q);

  // print it so we can see
  qcor::print_kernel(std::cout, measure_grov, q);

  // Run the kernel
  measure_grov(q);

  // print the results
  q.print();
}