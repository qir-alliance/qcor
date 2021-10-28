#include "qcor_qsim.hpp"

// High-level usage of Model Builder for typical VQE problem.

// Compile and run with:
/// $ qcor -qpu qpp VqeWithAnsatzCircuit.cpp
/// $ ./a.out

// Define a fixed ansatz as a QCOR kernel
__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  auto exponent_op = X(0) * Y(1) - Y(0) * X(1);
  exp_i_theta(q, theta, exponent_op);
}

int main(int argc, char **argv) {
  // Create the Deuteron Hamiltonian
  auto H = 5.907 - 2.1433 * X(0) * X(1) - 2.143 * Y(0) * Y(1) + 0.21829 * Z(0) -
           6.125 * Z(1);
  const auto num_qubits = 2;
  const auto num_params = 1;
  auto problemModel =
      QuaSiMo::ModelFactory::createModel(ansatz, H, num_qubits, num_params);
  auto optimizer = createOptimizer("nlopt");
  // Instantiate a VQE workflow with the nlopt optimizer
  auto workflow = QuaSiMo::getWorkflow("vqe", {{"optimizer", optimizer}});

  // Result should contain the ground-state energy along with the optimal
  // parameters.
  auto result = workflow->execute(problemModel);

  const auto energy = result.get<double>("energy");
  std::cout << "Ground-state energy = " << energy << "\n";
  return 0;
}