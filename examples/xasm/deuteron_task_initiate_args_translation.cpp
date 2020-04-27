#include "qcor.hpp"

__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  Ry(q[1], theta);
  CX(q[1], q[0]);
}

int main(int argc, char **argv) {
  // Allocate 2 qubits
  auto q = qalloc(2);

  // Create the Deuteron Hamiltonian (Observable)
  auto H = qcor::createObservable(
      "5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1");

  // Create the ObjectiveFunction, here we want to run VQE
  // need to provide ansatz and the Observable
  auto objective = qcor::createObjectiveFunction("vqe", ansatz, H);

  // Create the Optimizer
  auto optimizer = qcor::createOptimizer("nlopt");

  // Create mechanism for mapping Optimizer std::vector<double> parameters 
  // to the ObjectiveFunction variadic arguments of qreg and double
  auto args_translation = qcor::TranslationFunctor<qreg, double>(
      [&](const std::vector<double> x) { return std::make_tuple(q, x[0]); });

  // Call taskInitiate, kick off optimization of the give
  // functor dependent on the ObjectiveFunction, async call
  // Need to translate Optimizer std::vector<double> x params to Objective 
  // Function evaluation args qreg, double.
  auto handle = qcor::taskInitiate(objective, optimizer, args_translation, 1);

  // Go do other work...

  // Query results when ready.
  auto results = qcor::sync(handle);

  // Print the optimal value.
  printf("<H> = %f\n", results.opt_val);
}
