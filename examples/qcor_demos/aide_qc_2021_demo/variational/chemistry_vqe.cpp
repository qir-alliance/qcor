
// Demonstrate quick hamiltonian generation
// transformations on hamiltonians with JW
// Optimization to ground state energy

// Then demonstrate qubit-tapering transform

__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  X(q[2]);
  compute {
    Rx(q[0], constants::pi / 2);
    for (auto i : range(3)) H(q[i + 1]);
    for (auto i : range(3)) {
      X::ctrl(q[i], q[i + 1]);
    }
  }
  action { Rz(q[3], theta); }
}

__qpu__ void exp_ansatz(qreg q, double theta) {
  X(q[0]);
  X(q[2]);
  auto arg = X(0) * X(1) * X(2) * Y(3);
  exp_i_theta(q, theta, arg);
}

__qpu__ void one_qubit_ansatz(qreg qq, double theta, double phi) {
  auto q = qq.head();
  Ry(q, theta);
  Rz(q, phi);
}

int main() {
  std::string h2_geom = R"#(H  0.000000   0.0      0.0
H   0.0        0.0  .7474)#";
  auto H =
      createOperator("pyscf", {{"basis", "sto-3g"}, {"geometry", h2_geom}});

  // Can programmatically run transformations
  H = operatorTransform("jw", H);

  OptFunction opt_function(
      [&](std::vector<double> x) {
        return exp_ansatz::observe(H, qalloc(4), x[0]);
      },
      1);

  auto optimizer = createOptimizer("nlopt");
  auto [ground_energy, opt_params] = optimizer->optimize(opt_function);
  print("Energy: ", ground_energy);

  H = operatorTransform("qubit-tapering", H);

  OptFunction one_qubit_opt_function(
      [&](std::vector<double> x) {
        return one_qubit_ansatz::observe(H, qalloc(1), x[0], x[1]);
      },
      2);

  optimizer = createOptimizer("skquant");

  auto [ground_energy2, opt_params2] =
      optimizer->optimize(one_qubit_opt_function);
  print("Energy: ", ground_energy2);

  return 0;
}