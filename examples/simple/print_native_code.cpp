__qpu__ void bell(qreg q) {
  H(q[0]);
  CX(q[0], q[1]);
  Measure(q);
}

int main() {
  auto q = qalloc(2);
  // print_native_code for QuantumKernel
  bell::print_native_code(q);

  auto bell_lambda = qpu_lambda([](qreg q) {
    H(q[0]);
    X::ctrl(q[0], q[1]);
    Measure(q);
  });

  // print_native_code for qpu_lambda
  bell_lambda.print_native_code(q);


  using BellSignature = KernelSignature<qreg>;
  BellSignature callable(bell);
  // print_native_code for KernelSignature
  callable.print_native_code(q);
}
