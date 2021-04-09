using GroverPhaseOracle = KernelSignature<qreg>;

__qpu__ void reflect_about_uniform(qreg q) {
  compute {
    H(q);
    X(q);
  } action {
    std::vector<qubit> ctrl_qubits;
    for (int i = 0; i < q.size() - 1; i++) {
      std::cout << "adding qubit " << q[i].second << "\n";
      ctrl_qubits.push_back(q[i]);
    }
    auto last_qubit = q[2];
    Z::ctrl(ctrl_qubits, last_qubit);
  }

  return;
}

__qpu__ void run_grover(qreg q, GroverPhaseOracle oracle,
                        const int iterations) {
  H(q);
  for (int i = 0; i < iterations; i++) {
    oracle(q);
    reflect_about_uniform(q);
  }

  Measure(q);
}

__qpu__ void oracle(qreg q) {
    CZ(q[0], q[2]);
    CZ(q[1], q[2]);
}

__qpu__ void ccz(qreg q) {
    Z::ctrl({q[0], q[1]}, q[2]);
}

int main() {
    auto q = qalloc(3);
    run_grover(q, oracle, 1);
    q.print();
    run_grover::print_kernel(q, oracle, 1);

    auto m = ccz::as_unitary_matrix(q);
    std::cout << m << "\n";
    std::cout << run_grover::openqasm(q, oracle, 1) << "\n";
}