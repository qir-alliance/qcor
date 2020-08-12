#include <qalloc>

// Add a gate b/w q[0] and q[4] which do not connect directly
// to verify the placement algorithm.
__qpu__ void test_xasm(qreg q) {
  using qcor::xasm;
  H(q[0]);
  CX(q[0],q[4]);
  Measure(q[0]);
  Measure(q[4]);
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
  auto q = qalloc(5);
  {
    class test_xasm t(q);
    t.optimize_only = true;
  }
  std::cout << "AFTER PLACEMENT: \n" << quantum::program->toString() << "\n";
}
