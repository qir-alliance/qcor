__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  Ry(q[1], theta);
  CX(q[1], q[0]);
}

int main() {
  // Create the Hamiltonian
  auto H = -2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1) + 5.907;

  auto optimizer = createOptimizer("nlopt");
  ObjectiveFunction opt_function(
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto q = qalloc(2);
        auto exp = ansatz::observe(H, q, x[0]);
        print("<E(", x[0], ") = ", exp);
        return exp;
      },
      1);

  auto [energy, opt_params] = optimizer->optimize(opt_function);
  print("Min energy = ", energy);
}