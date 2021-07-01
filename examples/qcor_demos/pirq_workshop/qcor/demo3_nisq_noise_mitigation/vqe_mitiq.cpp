/// vqe_mitiq.cpp: Variational algorithm with noise mitigation
/// Compile:
/// $ qcor -qpu aer[noise-model:noise_model.json] -shots 8192 -em mitiq-zne vqe_mitiq.cpp

__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  Ry(q[1], theta);
  CX(q[1], q[0]);
}

int main(int argc, char *argv[]) {
  // Define the Deuteron hamiltonian
  auto H = -2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1);

  auto angles = qcor::linspace(-M_PI, M_PI, 10);
  std::vector<double> energy;
  for (const auto &angle : angles) {
    auto q = qalloc(2);
    energy.emplace_back(ansatz::observe(H, q, angle));
  }

  for (int i = 0; i < angles.size(); ++i) {
    print("Theta=", angles[i], "; Energy =", energy[i]);
  }
}