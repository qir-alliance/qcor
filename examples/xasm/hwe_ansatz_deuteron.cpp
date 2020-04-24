#include "qcor.hpp"

__qpu__ void ansatz(qreg q, std::vector<double> x) {
  for (int i = 0; i < 2; i++) {
    Rx(q[i], x[i]);
    Rz(q[i], x[2 + i]);
  }
  CX(q[1], q[0]);
  for (int i = 0; i < 2; i++) {
    Rx(q[i], x[i + 4]);
    Rz(q[i], x[i + 4 + 2]);
    Rx(q[i], x[i + 4 + 4]);
  }
}

int main(int argc, char **argv) {
  // Allocate 2 qubits
  qreg q = qalloc(2);

  // Create the Deuteron Hamiltonian (Observable)
  auto H = qcor::createObservable(
      "5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1");

  // Create the ObjectiveFunction, here we want to run VQE
  // need to provide ansatz and the Observable
  // Must also provide initial params for ansatz (under the hood, uses
  // variadic template)
  auto objective = qcor::createObjectiveFunction("vqe", ansatz, H);


  qcor::OptFunction f(
      [&](const std::vector<double> &x, std::vector<double> &grad) {
        return (*objective)(q, x);
      },
      10);

  auto optimizer = qcor::createOptimizer("nlopt");
  auto results = optimizer->optimize(f);

  // Print the result
  printf("vqe energy = %f\n", results.first);

  return 0;
}
