__qpu__ void ccnot(qreg q) {
  for (int i = 0; i < 8; ++i) {
    print("Initial state:", i);
    for (int j = 0; j < 3; ++j) {
      // set initial state
      if (i & (1 << j)) {
        X(q[j]);
      }
    }
    // CCNOT
    X::ctrl({q[0], q[1]}, q[2]);
    print("Measure:");
    for (int i = 0; i < q.size(); ++i) {
      if (Measure(q[i])) {
        print(1);
        X(q[i]);
      } else {
        print(0);
      }
    }
  }
}

int main() {
  // allocate 3 qubits
  auto q = qalloc(3);
  // Run the unitary evolution.
  ccnot(q);
}