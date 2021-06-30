// trotter_decompose.cpp: Demonstrate runtime optimization passes (acting on XACC IR)
// Simplify Trotter circuit with many iterations to a shorter form
// $qcor -opt 1 -print-opt-stats -print-final-submission trotter_decompose.cpp
__qpu__ void trotter_evolve(qreg q, Operator hamiltonian, double dt, int nbSteps) {
  for (int i = 0; i < nbSteps; ++i) {
    exp_i_theta(q, dt, hamiltonian);
  }
}

int main() {
  auto qbits = qalloc(2);
  auto H = X(0) * X(1);
  double dt = 0.01;
  int nbSteps = 100;
  trotter_evolve(qbits, H, dt, nbSteps);
}