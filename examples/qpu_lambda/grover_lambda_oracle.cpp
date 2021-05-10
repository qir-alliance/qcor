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
  set_shots(1024);

  const int N = 3;

  // Write the oracle as a quantum lambda function
  auto oracle = qpu_lambda([](qreg q) {
      print("hey from oracle: ", N);
      CZ(q[0], q[2]);
      CZ(q[1], q[2]);
  }, N);

  // Allocate some qubits
  auto q = qalloc(N);
  oracle.print_kernel(q);
  int iterations = 1;
  // Call grover given the oracle and n iterations
  run_grover(q, oracle, iterations);
  // print the histogram
  q.print();

  // Grover lambda:
  // amplification lambda
  auto amplification_lambda = qpu_lambda([](qreg q) {
    print("hey from amplification_lambda");
    compute {
      H(q);
      X(q);
    }
    action {
      auto ctrl_bits = q.head(q.size() - 1);
      auto last_qubit = q.tail();
      Z::ctrl(ctrl_bits, last_qubit);
    }
  });

  // Capture the grover lambda and iterations directly from the enclosing scope.
  auto grover_lambda = qpu_lambda([](qreg q) {
        H(q);
        for (int i = 0; i < iterations; i++) {
          oracle(q);
          amplification_lambda(q);
        }

        Measure(q);
      }, oracle, iterations, amplification_lambda);

  auto q_lambda = qalloc(N);

  std::cout << "Lamda result:\n";
  grover_lambda.print_kernel(q_lambda);
  grover_lambda(q_lambda);
  q_lambda.print();
}