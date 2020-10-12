#include "qcor_qsim.hpp"

// Solving the ground-state energy of a Hamiltonian operator by the iterative
// QPE procedure.

// Compile and run with:
/// $ qcor -qpu qpp IterativeQpeWorkflow.cpp
/// $ ./a.out

__qpu__ void eigen_state_prep(qreg q) {
  auto theta = 0.297113;
  X(q[0]);
  auto exponent_op = X(0) * Y(1) - Y(0) * X(1);
  exp_i_theta(q, theta, exponent_op);
}

int main(int argc, char **argv) {
  // Create Hamiltonian
  auto H = 5.907 - 2.1433 * X(0) * X(1) - 2.143 * Y(0) * Y(1) + 0.21829 * Z(0) -
           6.125 * Z(1);
  auto problemModel =
      qsim::ModelBuilder::createModel(eigen_state_prep, H, 2, 0);

  // Instantiate an IQPE workflow.
  auto workflow = qsim::getWorkflow("iqpe", {{"iterations", 8}});

  // Result should contain the observable expectation value along Trotter steps.
  auto result = workflow->execute(problemModel);
  return 0;
}