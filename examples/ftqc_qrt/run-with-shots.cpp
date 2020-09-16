#include <qalloc>
// Compile with: qcor -qpu qpp -qrt ftqc -shots 1024 run-with-shots.cpp

// Define sub-kernel to print out FTQC execution
__qpu__ void h_gate(qreg q) { 
  std::cout << "Run H\n";
  H(q[0]); 
}
__qpu__ void cx_gate(qreg q) { 
  std::cout << "Run CNOT\n";
  CX(q[0], q[1]);
}

__qpu__ void bell(qreg q) {
  using qcor::xasm;
  h_gate(q);
  cx_gate(q);
  const bool q0Result = Measure(q[0]);
  const bool q1Result = Measure(q[1]);
  if (q0Result == q1Result) {
    std::cout << " Matched!\n";
  } else {
    std::cout << "NOT Matched!\n";
  }
}

int main() {
  auto q = qalloc(2);
  bell(q);
  q.print();
}
