// trotter_decompose.cpp: Demonstrate runtime optimization passes (acting on XACC IR)
// (with runtime information -> resolved/flattened circuit -> optimization)
// Simplify Trotter circuit with many iterations to a shorter form
// $qcor -opt 2 -print-opt-stats -print-final-submission trotter_decompose.cpp
// Run:
// ./a.out -dt xxx -steps yyy
__qpu__ void trotter_evolve(qreg q, Operator hamiltonian, double dt, int nbSteps) {
  for (int i = 0; i < nbSteps; ++i) {
    exp_i_theta(q, dt, hamiltonian);
  }
}

int main(int argc, char **argv) {
  auto qbits = qalloc(2);
  // Multi-term Hamiltonian
  auto H = X(0) * X(1) + Z(0) + Z(1);

  // Parse user-supplied params (if any)
  // dt: Trotter time step size
  // steps: number of steps
  argparse::ArgumentParser program(argv[0]);
  program.add_argument("-dt").default_value(0.01).action(
      [](const std::string &value) { return std::stod(value); });
  program.add_argument("-steps").default_value(100).action(
      [](const std::string &value) { return std::stoi(value); });
  program.parse_args(argc, argv);

  // Get runtime params (if any)
  const double dt = program.get<double>("-dt");
  const int nbSteps = program.get<int>("-steps");

  std::cout << "Trotter for dt = " << dt << "; steps = " << nbSteps << "\n";
  trotter_evolve(qbits, H, dt, nbSteps);
}