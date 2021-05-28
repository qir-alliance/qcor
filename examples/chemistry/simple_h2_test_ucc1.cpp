__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  X(q[2]);
  compute {
    Rx(q[0], constants::pi / 2);
    for (auto i : range(3)) H(q[i + 1]);
    for (auto i : range(3)) {
      CX(q[i], q[i + 1]);
    }
  }
  action { Rz(q[3], theta); }
}

int main() {

  std::string h2_geom = R"#(H  0.000000   0.0      0.0
H   0.0        0.0  .7474)#";
  auto H =
      createOperator("pyscf", {{"basis", "sto-3g"}, {"geometry", h2_geom}});

  OptFunction opt_function(
      [&](std::vector<double> x) {
        return ansatz::observe(H, qalloc(4), x[0]);
      },
      1);
  
   auto [energy, opt_params] = createOptimizer("nlopt")->optimize(opt_function);
   print(energy);
}