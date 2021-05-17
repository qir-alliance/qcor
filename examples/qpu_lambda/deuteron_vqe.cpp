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

  OptFunction opt_function(
      [&](std::vector<double> x) { return ansatz.observe(H, qalloc(2), x[0]); },
      1);

  auto optimizer = createOptimizer("nlopt");
  auto [ground_energy, opt_params] = optimizer->optimize(opt_function);
  print("Energy: ", ground_energy);
  qcor_expect(std::abs(ground_energy + 1.74886) < 0.1);
}