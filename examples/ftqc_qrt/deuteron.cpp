#include "ftqc/vqe.hpp"

// Compile with:
// qcor -qpu qpp -qrt ftqc -I<qcor/examples/shared> deuteron.cpp 

__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  Ry(q[1], theta);
  CX(q[1], q[0]);
}

__qpu__ void test(qreg q, double theta, std::vector<qcor::PauliOperator> bases) {
  ansatz(q, theta);
  MeasureP(q, bases);
}

int main(int argc, char **argv) {
  // Allocate 2 qubits
  auto q = qalloc(2);

  // auto H = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
  //          6.125 * Z(1);

  std::vector<qcor::PauliOperator> ops { X(0), X(1)};
  test(q, M_PI_4, ops);
  q.print();
}
