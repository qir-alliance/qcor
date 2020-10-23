#include "qcor_qsim.hpp"

// High-level usage of Model Builder for typical VQE problem.

// Compile and run with:
// (need shots to do verification)
/// $ qcor -qpu qpp -shots 8192 VqeWithVerifiedQpeMitigation.cpp
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
  // QPE with verification
  auto costEvaluator = qsim::getObjEvaluator(H, "qpe", {{"verified", true}});

  // QPE no verification
  // auto costEvaluator = qsim::getObjEvaluator(H, "qpe");

  // Default (tomography-based)
  // auto costEvaluator = qsim::getObjEvaluator(H);

  auto problemModel =
      qsim::ModelBuilder::createModel(ansatz, H, num_qubits, num_params);

  // Initial parameters: we evaluate the error at single data points.
  const std::vector<double> init_params{0.297113};
  auto optimizer = createOptimizer(
      "nlopt", {{"nlopt-maxeval", 1}, {"initial-parameters", init_params}});
  // Instantiate a VQE workflow with the nlopt optimizer
  auto workflow = qsim::getWorkflow(
      "vqe", {{"optimizer", optimizer}, {"evaluator", costEvaluator}});

  // Result should contain the observable expectation value along Trotter steps.
  auto result = workflow->execute(problemModel);

  const auto energy = result.get<double>("energy");
  std::cout << "Ground-state energy = " << energy << "\n";
  return 0;
}