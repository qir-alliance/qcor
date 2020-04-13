#include "qcor.hpp"

__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  Ry(q[1], theta);
  CX(q[1],q[0]);
}

int main(int argc, char **argv) {
  // Allocate 2 qubits
  auto q = qalloc(2);

  // Create the Deuteron Hamiltonian
  auto H = qcor::getObservable(
      "5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1");

  auto objective = qcor::createObjectiveFunction("vqe", ansatz, H, q, 0.0);

  auto energy = (*objective)(q, .59);

  // Print the result
  printf("vqe energy = %f\n", energy);

  q.print();
  
  return 0;
}
