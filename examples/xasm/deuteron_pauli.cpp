#include "qcor.hpp"

__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  Ry(q[1], theta);
  CX(q[1], q[0]);
}
int main(int argc, char **argv) {
  auto q = qalloc(2);

  auto H = 5.907 - 2.1433 * qcor::X(0) * qcor::X(1) -
           2.1433 * qcor::Y(0) * qcor::Y(1) + .21829 * qcor::Z(0) -
           6.125 * qcor::Z(1);

//   auto ansatz_exponent = qcor::X(0) * qcor::Y(1) - qcor::Y(0) * qcor::X(1);

  qcor::OptFunction opt_func(
      [&](const std::vector<double> &x, std::vector<double> &grad) -> double {
        // Affect Observable observation and evaluate at given ansatz parameters
        auto e = qcor::observe(ansatz, H, q, x[0]);

        // Need to clean the current qubit register
        q.reset();
        return e;
      },
      1);

  auto optimizer = qcor::createOptimizer("nlopt");

  auto result = optimizer->optimize(opt_func);

  printf("energy = %f\n", result.first);

  return 0;
}