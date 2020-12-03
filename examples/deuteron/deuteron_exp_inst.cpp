
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

  // Create the objective function
  auto objective = createObjectiveFunction(ansatz, H, q, 1);
  
  // Create a qcor Optimizer
  auto optimizer = createOptimizer("nlopt");

  // Optimize the above function
  auto [optval, opt_params] = optimizer->optimize(*objective.get());

  // Print the result
  printf("energy = %f\n", optval);

  return 0;
}
