using GroverPhaseOracle = KernelSignature<qreg>;

__qpu__ void reflect_about_uniform(qreg q) {
  compute {
    H(q);
    X(q);
  } action {
    // we have N qubits, get the first N-1 as the 
    // ctrl qubits, and the last one for the 
    // ctrl-ctrl-...-ctrl-z operation qubit
    auto ctrl_qubits = q.head(q.size()-1);
    auto last_qubit = q.tail();
    Z::ctrl(ctrl_qubits, last_qubit);
  }
}

__qpu__ void run_grover(qreg q, GroverPhaseOracle oracle,
                        const int iterations) {
  // Put them all in a superposition
  H(q);

  // Iteratively apply the oracle then reflect
  for (int i = 0; i < iterations; i++) {
    oracle(q);
    reflect_about_uniform(q);
  }

  // Measure all qubits
  Measure(q);
}

__qpu__ void oracle(qreg q) {
    CZ(q[0], q[2]);
    CZ(q[1], q[2]);
}

int main(int argc, char** argv) {
    auto q = qalloc(3);
    run_grover(q, oracle, 1);
    for (auto [bits, count] : q.counts()) {
      print(bits, ":", count);
    }
}
