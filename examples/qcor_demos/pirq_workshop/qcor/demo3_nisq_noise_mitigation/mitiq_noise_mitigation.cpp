/// mitiq_noise_mitigation.cpp: Run error mitigation with mitiq ZNE (zero-noise extrapolation)
/// Compile:
/// $ qcor -qpu aer[noise-model:noise_model.json] -shots 8192 -em mitiq-zne
/// mitiq_noise_mitigation.cpp Execute: specify the number of repeating CNOT
/// ./a.out N

// Repeating CNOT's to evaluate noise mitigation.
// This is a do-nothing circuit: qubits should return to the |00> state.
__qpu__ void noisy_zero(qreg q, int cx_count) {
  H(q);
  for (int i = 0; i < cx_count; i++) {
    X::ctrl(q[0], q[1]);
  }
  H(q);
  Measure(q);
}

int main(int argc, char *argv[]) {
  // Default depth
  int CX_depth = 0;

  // Parse number of CX cycles:
  if (argc == 2) {
    CX_depth = std::stoi(argv[1]);
  }
  qreg q = qalloc(2);
  // noisy_zero::print_kernel(q, CX_depth);
  noisy_zero(q, CX_depth);
  // q.print();
  std::cout << "CX depth: " << CX_depth << "; Expectation: " << q.exp_val_z() << "\n";
}