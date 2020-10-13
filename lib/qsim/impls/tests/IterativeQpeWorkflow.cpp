#include "qcor_qsim.hpp"

// Solving the ground-state energy of a Hamiltonian operator by the iterative
// QPE procedure.

// Compile and run with:
/// $ qcor -qpu qpp IterativeQpeWorkflow.cpp
/// $ ./a.out
// Ansatz to bring the state into an eigenvector state of the Hamiltonian.
__qpu__ void eigen_state_prep(qreg q) {
  using qcor::openqasm;
  u3(pi / 4, 0, 0) q[0];
}

int main(int argc, char **argv) {
  // Create Hamiltonian:
  // Important: this must be a weighted pauli operator list.
  // TODO: add function to convert arbitrary Hamiltonian to the weighted form.
  auto H = 0.5 + 0.25 * X(0) + 0.25 * Z(0);
  auto problemModel =
      qsim::ModelBuilder::createModel(eigen_state_prep, H, 1, 0);

  // Instantiate an IQPE workflow.
  auto workflow =
      qsim::getWorkflow("iqpe", {{"time-steps", 4}, {"iterations", 4}});

  auto result = workflow->execute(problemModel);
  const double phaseValue = result.get<double>("phase");
  std::cout << "Final phase = " << phaseValue << "\n";
  return 0;
}