// Demonstrate simple optimizations
// simple rotation gate mergers
// h rx ry h

// Show opt level 1, should have gates remaining, but 
// they are == I, then show opt level 2

__qpu__ void nothing(qreg qbits, const int n, double x) {
  auto q = qbits.head();
  auto r = qbits.tail();

  for (int i = 0; i < n; i++) {
    H(q);
  }

  // This is Z Z = I
  Z(q);
  H(q);
  X(q);
  H(q);

  // Should be I if Rx on 0.0
  H(q);
  CX(q, r);
  Rx(q, x);
  X::ctrl(q, r);
  H(q);
}

int main() {
  // get a qubit
  auto qbits = qalloc(2);

  nothing(qbits, 100, 0.0);

  std::cout << nothing::as_unitary_matrix(qbits, 100, 0.0) << "\n";
}