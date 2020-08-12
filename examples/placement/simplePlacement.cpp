#include <qalloc>

// Creates a 4-qubit GHZ state |0000> + |1111>
// which, on ibmq_ourense, will require mapping
// to fit its connectivity graph  
__qpu__ void test_xasm(qreg q) {
  using qcor::xasm;
  H(q[0]);
  CX(q[0],q[1]);
  CX(q[0],q[2]);
  CX(q[0],q[3]);
  Measure(q[0]);
  Measure(q[1]);
  Measure(q[2]);
  Measure(q[3]);
}

// Example: using ibmq_ourense (5 qubits) which has 
// the following connectivity graph:
// 0 -- 1 -- 2
//      |
//      3
//      |
//      4
// Compile: qcor -qpu aer:ibmq_ourense simplePlacement.cpp 
// Make sure to have a valid ~/.ibm_config file.
// Example: 
// key: YOUR_API_KEY
// hub:ibm-q
// group:open
// project:main
// url: https://quantumexperience.ng.bluemix.net

int main() {
  // This circuit requires 4 qubits.
  auto q = qalloc(4);
  test_xasm(q);
  // Expect: ~50-50 for "0000" and "1111"
  // (plus some variations due to noise)
  q.print();
}
