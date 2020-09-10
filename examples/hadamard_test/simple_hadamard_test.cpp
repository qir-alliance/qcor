#include <iomanip>

__qpu__ void x_gate(qreg q) { X(q[1]); }
__qpu__ void h_gate(qreg q) { H(q[1]); }

__qpu__ void htest(qreg q) {
  // Create the superposition on the first qubit
  H(q[0]);

  // create the |1> on the second qubit
  x_gate(q);

  // create superposition on second qubit
  // h_gate(q);

  // apply ctrl-U
  x_gate::ctrl(q[0], q);

  // add the last hadamard
  H(q[0]);

  // measure
  Measure(q[0]);
}

int main() {
  auto q = qalloc(2);
  htest(q);
  q.print();
  auto count1 = q.counts().find("1")->second;
  auto count2 = q.counts().find("0")->second;
  std::cout << "<X> = " << std::setprecision(12)
            << std::fabs((count1 - count2) / (double)(count1 + count2)) << "\n";
}