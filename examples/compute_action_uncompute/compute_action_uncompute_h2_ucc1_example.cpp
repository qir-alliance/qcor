__qpu__ void hf(qreg q) {
  X(q[0]);
  X(q[2]);
}
__qpu__ void ucc1(qreg q, double x) {
  // The following syntax gives us
  // U V Udag functionality. This provides
  // the familia ProjectQ / Q# / etc.
  // compute - action - uncompute
  compute {
    Rx(q[0], constants::pi / 2);
    for (auto i : range(3)) H(q[i + 1]);
    for (auto i : range(3)) {
      CX(q[i], q[i + 1]);
    }
  }
  action { Rz(q[3], x); }
}

__qpu__ void ansatz(qreg q, double x) {
  hf(q);
  ucc1(q, x);
}

__qpu__ void test_ctrl(qreg q, double d) { ucc1::ctrl(q[4], q, d); }

int main() {
  auto H = createOperator(
      "pyscf", {{"basis", "sto-3g"}, {"geometry", R"#(H  0.000000   0.0      0.0
H   0.0        0.0  .7474)#"}});

  auto objective = createObjectiveFunction(ansatz, H, 1);
  auto optimizer = createOptimizer("nlopt", {{"maxeval", 20}});

  // Optimize the above function
  auto [optval, opt_params] = optimizer->optimize(*objective.get());

  // Print the result
  printf("energy = %f\n", optval);

  ansatz::print_kernel(std::cout, qalloc(4), 2.2);

  // Should only see CRZ, U, Udag should be uncontrolled
  auto qq = qalloc(5);
  test_ctrl::print_kernel(std::cout, qq, 2.2);
}