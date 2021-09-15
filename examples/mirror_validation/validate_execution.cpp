// Compile to run with validation mode: 
// Qpp (noiseless)
// qcor -validate validate_execution.cpp -shots 1024
// Aer (noisy)
// qcor -validate validate_execution.cpp -shots 1024 -qpu aer[noise-model:noise_model.json]
__qpu__ void noisy_zero(qreg q, int cx_count) {
  H(q);
  for (int i = 0; i < cx_count; i++) {
    X::ctrl(q[0], q[1]);
  }
  H(q);
  Measure(q);
}

int main() {
  // qcor::set_verbose(true);
  // On the noisy simulator, the validation will be successful for 1 cycle
  // but will probably fail when running with 10 cycles.
  const int nb_cycles = 1;
  // const int nb_cycles = 10;
  auto q = qalloc(2);
  noisy_zero(q, nb_cycles);
  q.print();
}