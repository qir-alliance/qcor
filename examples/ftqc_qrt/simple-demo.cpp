#include <qalloc>
// Compile with: qcor -qpu qpp -qrt ftqc simple-demo.cpp
// Execute: ./a.out
// We should get the print out conditioned by the measurement.
// If not using the "ftqc" QRT, this will cause errors since the Measure results
// are not available yet.

__qpu__ void bell(qreg q) {
  using qcor::xasm;
  H(q[0]);
  CX(q[0], q[1]);
  for (int i = 0; i < 2; i++)
    X(q[i]);
  
  Measure(q[0]);
  
  if (q.cReg(0)) {
    std::cout << "Q0 = 1 !\n";
  } 
  
  Measure(q[1]);
  if (q.cReg(1)) {
    std::cout << "Q1 = 1 !\n";
  } 
}

int main() {
  qcor::set_verbose(true);
  auto q = qalloc(2);
  bell(q);
}
