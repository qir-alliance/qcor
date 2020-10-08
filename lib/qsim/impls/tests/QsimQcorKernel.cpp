#include "qcor_qsim.hpp"

__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  auto exponent_op = X(0) * Y(1) - Y(0) * X(1);
  exp_i_theta(q, theta, exponent_op);
}

int main(int argc, char **argv) {
  // Allocate 2 qubits
  auto q = qalloc(2);

  // Create the Deuteron Hamiltonian
  auto H = createObservable(
      "5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1");

  auto problemModel = ModelBuilder::createModel(ansatz, H.get(), q.size(), 1);
  auto optimizer = createOptimizer("nlopt");

  auto workflow = qcor::getWorkflow("vqe", {{"optimizer", optimizer}});

  // Result should contain the observable expectation value along Trotter steps.
  auto result = workflow->execute(problemModel);

  return 0;
}