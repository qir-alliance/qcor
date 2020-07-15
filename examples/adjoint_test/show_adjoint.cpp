#include "qcor.hpp"

__qpu__ void measure_qbits(qreg q) {
  for (int i = 0; i < 2; i++) {
    Measure(q[i]);
  }
}

__qpu__ void quantum_kernel(qreg q, double x) {
    X(q[0]);
    Ry(q[1], x);
    CX(q[1],q[0]);
}

__qpu__ void z0z1(qreg q, double x) {
    quantum_kernel(q, x);
    measure_qbits(q);
}

__qpu__ void check_adjoint(qreg q, double x) {
    quantum_kernel(q,x);
    quantum_kernel::adjoint(q,x);
    measure_qbits(q);
}

int main() {
  auto q = qalloc(2);

  quantum_kernel(q, 2.2);

  q.print();

  auto r = qalloc(2);

  z0z1(r, 2.2);
  r.print();

  auto v = qalloc(2);

  check_adjoint(v, 2.2);
  v.print();

}