#include "qcor.hpp"

__qpu__ void ansatz(qreg q, double theta, std::shared_ptr<xacc::Observable> H) {
  X(q[0]);
  exp_i_theta(q, theta, H);
}

int main(int argc, char **argv) {
  // Allocate 2 qubits
  auto q = qalloc(2);

  // Create the Deuteron Hamiltonian
  auto H = qcor::getObservable(
      "5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1");

  // Create the Ansatz exponent Operator
  auto ansatz_exponent = qcor::getObservable("X0 Y1 - Y0 X1");

  // Create an objective function to optimize
  // Each call evaluates E(theta) = <ansatz(theta) | H | ansatz(theta)>
  xacc::OptFunction opt_func(
      [&](const std::vector<double> &x, std::vector<double> &grad) -> double {
        
        // Affect Observable observation and evaluate at given ansatz parameters
        auto e = qcor::observe(ansatz, H, q, x[0], ansatz_exponent);

        // Need to clean the current qubit register
        q.reset();
        return e;
      },
      1);

  // Create a qcor Optimizer
  auto optimizer = qcor::getOptimizer();

  // Optimize the above function
  auto result = optimizer->optimize(opt_func);

  // Print the result
  printf("energy = %f\n", result.first);

  return 0;
}
