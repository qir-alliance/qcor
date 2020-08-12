#include <qalloc>

// Create a multi-qubit entangled state 
__qpu__ void entangleQubits(qreg q) {
  H(q[0]);
  for (int i = 1; i < q.size(); i++) {
    CX(q[0],q[i]);
  }
  for (int i = 0; i < q.size(); i++) {
    Measure(q[i]);
  }
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
  entangleQubits(q);
  // Expect: ~50-50 for "0000" and "1111"
  // (plus some variations due to noise)
  q.print();
}
