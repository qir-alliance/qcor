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

__qpu__ void measured_grover_5(qreg q) {
    grover_5(q);
    for (int i = 0; i < q.size(); i++) {
        Measure(q[i]);
    }
}

int main() {

  // Allocate the qubits
  auto q = qalloc(9);

  measured_grover_5::print_kernel(std::cout, q);

  // Run the kernel
  measured_grover_5(q);

  // print the results
  q.print();
}