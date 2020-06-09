#include "qcor.hpp"

__qpu__ void call_from_f(qreg q) {
  H(q[0]);
  H(q[0]);
  Measure(q[0]);
}

__qpu__ void quantum_kernel_me(qreg q, double angle) {
  H(q[0]);
  H(q[0]);
  H(q[0]);

  Ry(q[0], angle);

  CX(q[0], q[1]);
  CX(q[0], q[1]);
  CX(q[0], q[1]);
  call_from_f(q);
  Measure(q[1]);
}

int main() {

  for (double &d : std::vector<double>{1.1, 2.2, 3.3}) {
    auto q = qalloc(2);

    quantum_kernel_me(q, d);

    q.print();
    std::cout << "compiler optimized:\n";
    qcor::print_kernel(std::cout, quantum_kernel_me, q, d);
  }
}