#include "qcor.hpp"

int main() {
  // Create the Hamiltonian
  auto H = -2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1) + 5.907;

  auto ansatz = qpu_lambda([](qreg q, double x) {
    X(q[0]);
    Ry(q[1], x);
    CX(q[1], q[0]);
  });

  auto q = qalloc(2);
  auto args_translator = std::make_shared<ArgsTranslator<qreg, double>>(
      [&](const std::vector<double> x) { return std::make_tuple(q, x[0]); });
  auto objective = createObjectiveFunction(ansatz, args_translator, H, q, 1);

  // Create a qcor Optimizer
  auto optimizer = createOptimizer("nlopt");

  // Optimize the above function
  auto [optval, opt_params] = optimizer->optimize(*objective.get());
}