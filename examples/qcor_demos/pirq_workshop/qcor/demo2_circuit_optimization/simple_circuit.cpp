// simple_circuit.cpp: Demonstrate runtime optimization passes (acting on XACC IR)
// $qcor -opt 1 -print-opt-stats simple_circuit.cpp -print-final-submission
__qpu__ void do_nothing(qubit q) {
  // This is Z Z = I
  // (H - X - H == Z)
  Z(q);
  H(q);
  X(q);
  H(q);
}

int main() {
  // get a qubit
  auto qbits = qalloc(1);
  do_nothing(qbits[0]);
  std::cout << "Unitary matrix: \n" << do_nothing::as_unitary_matrix(qbits[0]) << "\n";
}