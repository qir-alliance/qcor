#include "qcor.hpp"
// Create a general grover search algorithm.
// Let's create that marks 2 states
// Show figures Init - [Oracle - Amplification for i in iters] - Measure
// https://www.nature.com/articles/s41467-017-01904-7

// Show off kernel composition, common patterns, 
// functional programming (kernels taking other kernels)

using GroverPhaseOracle = KernelSignature<qreg>;

__qpu__ void amplification(qreg q) {
  // H q X q ctrl-ctrl-...-ctrl-Z H q Xq
  // compute - action - uncompute
  compute {
    H(q);
    X(q);
  }
  action {
    auto ctrl_bits = q.head(q.size() - 1);
    auto last_qubit = q.tail();
    Z::ctrl(ctrl_bits, last_qubit);
  }
}

__qpu__ void run_grover(qreg q, GroverPhaseOracle oracle,
                        const int iterations) {
  H(q);

  for (int i = 0; i < iterations; i++) {
    oracle(q);
    amplification(q);
  }

  Measure(q);
}

int main() {
  const int N = 3;

  // Write the oracle as a quantum lambda function
  auto oracle = qpu_lambda([](qreg q) {
      print("hey from oracle: ", N);
      CZ(q[0], q[2]);
      CZ(q[1], q[2]);
  }, N);

  // Allocate some qubits
  auto q = qalloc(N);

  // Call grover given the oracle and n iterations
  run_grover(q, oracle, 1);

  // print the histogram
  q.print();
}