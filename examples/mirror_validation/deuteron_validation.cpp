// Compile to run with validation mode: 
// Qpp (noiseless)
// qcor -validate deuteron_validation.cpp -shots 1024
// Aer (noisy)
// qcor -validate deuteron_validation.cpp -shots 1024 -qpu aer[noise-model:noise_model.json]
__qpu__ void deuteron(qreg q, double theta) {
  X(q[0]);
  Ry(q[1], theta);
  CNOT(q[1], q[0]);
  // for (int i = 0; i < 20; i++) {
  //   CNOT(q[1], q[0]);
  // }
  H(q[0]);
  H(q[1]);
  Measure(q[0]);
  Measure(q[1]);
}

int main() {
  qcor::set_verbose(true);
  const double angle = 0.297113;
  auto q = qalloc(2);
  deuteron(q, angle);
  q.print();
  std::cout << "<XX> = " << q.exp_val_z() << "\n";
}