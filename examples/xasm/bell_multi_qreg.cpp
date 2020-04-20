#include <qalloc>

// Define a multi-register kernel
__qpu__ void bell_multi(qreg q, qreg r) {
  H(q[0]);
  CX(q[0], q[1]);
  H(r[0]);
  CX(r[0], r[1]);
  Measure(q);
  Measure(r);
}

int main() {

  // Create two qubit registers, each size 2
  auto q = qalloc(2);
  auto r = qalloc(2);

  // Run the quantum kernel
  bell_multi(q, r);

  // dump the results
  q.print();
  r.print();
}