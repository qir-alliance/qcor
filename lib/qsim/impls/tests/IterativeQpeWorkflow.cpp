#include "qcor_qsim.hpp"

// Solving the ground-state energy of a Hamiltonian operator by the iterative
// QPE procedure.

// Compile and run with:
/// $ qcor -qpu qpp IterativeQpeWorkflow.cpp
/// $ ./a.out

int main(int argc, char **argv) {
  // Create the Deuteron Hamiltonian
  auto H = 5.907 - 2.1433 * X(0) * X(1) - 2.143 * Y(0) * Y(1) + 0.21829 * Z(0) -
           6.125 * Z(1);
  auto problemModel =
      qsim::ModelBuilder::createModel(&H);

  // Instantiate an IQPE workflow.
  auto workflow = qsim::getWorkflow("iqpe", {});

  // Result should contain the observable expectation value along Trotter steps.
  auto result = workflow->execute(problemModel);
  return 0;
}