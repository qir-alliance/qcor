__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  Ry(q[1], theta);
  CX(q[1], q[0]);
}

int main(int argc, char **argv) {
  // Allocate 2 qubits
  auto q = qalloc(2);

  // Programmer needs to set
  // the number of variational params
  auto n_variational_params = 1;

  // Create the Deuteron Hamiltonian
  auto H = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1);

  // Create the Optimizer.
  auto optimizer = createOptimizer("nlopt", {{"nlopt-optimizer", "l-bfgs"}});
  OptFunction opt_function(
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto q = qalloc(2);
        auto exp = ansatz::autograd(H, dx, q, x[0]);
        print("<E(", x[0], ") = ", exp);
        return exp;
      },
      n_variational_params);

  auto [energy, opt_params] = optimizer->optimize(opt_function);
  print(energy);
}
