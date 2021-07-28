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
  // Default params:
  double dt = 0.01;
  int nbSteps = 100;

  // Parse user-supplied params (if any)
  std::vector<std::string> arguments(argv + 1, argv + argc);
  for (int i = 0; i < arguments.size(); i++) {
    if (arguments[i] == "-dt") {
      dt = std::stod(arguments[i + 1]);
    }
    if (arguments[i] == "-steps") {
      nbSteps = std::stoi(arguments[i + 1]);
    }
  }
  std::cout << "Trotter for dt = " << dt << "; steps = " << nbSteps << "\n";
  trotter_evolve(qbits, H, dt, nbSteps);
}